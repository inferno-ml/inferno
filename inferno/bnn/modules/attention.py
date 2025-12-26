from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import params
from ... import bnn
from .bnn_mixin import BNNMixin

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class MultiheadAttention(BNNMixin, nn.Module):
    """Attention layer (with multiple heads).

    Multi-head (self-)attention layer with an optional attention mask, allowing a model to jointly attend
    to information from different representation subspaces. Consists of ``num_heads`` scaled dot-product
    attention modules, whose outputs are concatenated and then combined into an output sequence via a
    linear layer.

    The module supports nested or padded tensors and is inspired by the following
    [implementation](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html).

    :param embed_dim: Dimensionality of the inputs and outputs to the layer (i.e. dimensionality of the query embeddings).
    :param num_heads: Number of attention heads.
    :param dropout: Dropout probability; if greater than 0.0, dropout is applied.
    :param bias: Whether to add bias to query, key, value and output projections.
    :param kdim: Dimensionality of the key embeddings.
    :param vdim: Dimensionality of the value embeddings.
    :param embed_dim_out: Dimensionality of the output embeddings. If ``None``, set to ``embed_dim``.
    :param out_proj: Whether to include the output projection layer.
    :param cov: Covariance structure of the weights. Either a single covariance structure used in all
        linear projections, or a dictionary with keys ``k``, ``q``, ``v`` and ``out`` and values containing
        either covariance structures or ``None``.
    :param parametrization: The parametrization to use. Defines the initialization
        and learning rate scaling for the parameters of the module.
    :param device: Device on which to instantiate the parameters.
    :param dtype: Data type of the parameters.
    """

    # TODO: enable covariances that are block-diagonal by attention head

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: int | None = None,
        vdim: int | None = None,
        embed_dim_out: int | None = None,
        out_proj: bool = True,
        cov: (
            params.FactorizedCovariance | dict[params.FactorizedCovariance] | None
        ) = None,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(parametrization=parametrization)

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension is not divisible by num_heads.")
        self.head_dim = embed_dim // num_heads
        self.embed_dim_out = embed_dim_out if embed_dim_out is not None else embed_dim

        if cov is None:
            cov = {key: None for key in ["q", "k", "v", "out"]}
        elif isinstance(cov, params.FactorizedCovariance):
            cov = {key: copy.deepcopy(cov) for key in ["q", "k", "v", "out"]}

        self.q_proj = bnn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=bias,
            cov=cov["q"],
            parametrization=parametrization,
            **factory_kwargs,
        )
        self.k_proj = bnn.Linear(
            self.kdim,
            self.embed_dim,
            bias=bias,
            cov=cov["k"],
            parametrization=parametrization,
            **factory_kwargs,
        )
        self.v_proj = bnn.Linear(
            self.vdim,
            self.embed_dim,
            bias=bias,
            cov=cov["v"],
            parametrization=parametrization,
            **factory_kwargs,
        )
        self.out_proj = (
            bnn.Linear(
                self.embed_dim,
                self.embed_dim_out,
                bias=bias,
                cov=cov["out"],
                parametrization=parametrization,
                **factory_kwargs,
            )
            if out_proj
            else None
        )

    def forward(
        self,
        query: Float[Tensor, "*sample batch query_token embed_dim"],
        key: Float[Tensor, "*sample batch keyval_token embed_dim_k"] | None,
        value: Float[Tensor, "*sample batch token embed_dim_v"] | None,
        attn_mask: Float[Tensor, "batch query_token keyval_token"] | None = None,
        is_causal: bool = False,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample batch query_token embed_dim"]:
        """Computes scaled dot product attention on query, key and value tensors, using an optional attention mask.

        :param query: Query tensor / embeddings.
        :param key: Key tensor / embeddings.
        :param value: Value tensor / embeddings.
        :param attn_mask: Attention mask; shape must be broadcastable to the shape of attention weights.
                Two types of masks are supported.
                A boolean mask where a value of True indicates that the element *should* take part in attention.
                A float mask of the same type as query, key, value that is added to the attention score.
        :param is_causal: If set to true, the attention masking is a lower triangular matrix when the mask is a
                square matrix. The attention masking has the form of the upper left causal bias due to the alignment
                (see [``torch.nn.attention.bias.CausalBias``][]) when the mask is a non-square matrix.
                An error is thrown if both ``attn_mask`` and ``is_causal`` are set.
        :param sample_shape: Shape of samples. If None, runs a forward pass with just
            the mean parameters.
        :param generator: Random number generator.
        :param input_contains_samples: Whether the input already contains
            samples. If True, the input is assumed to have ``len(sample_shape)``
            many leading dimensions containing input samples (typically
            outputs from previous layers).
        :param parameter_samples: Dictionary of parameter samples. Used to pass
            sampled parameters to the module. Useful to jointly sample parameters
            of multiple layers.
        """
        # Step 1: Apply input projections
        if key is None:
            key = query
        if value is None:
            value = query

        query = self.q_proj(
            query,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        key = self.k_proj(
            key,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        value = self.v_proj(
            value,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )

        # Step 2: Split heads and prepare for scaled dot-product attention

        # reshape query, key, value to separate by head
        # (batch_size, seq_length_out, embed_dim_all_heads)
        # -> (batch_size, seq_length_out, num_heads, embed_dim_head)
        # -> (batch_size, num_heads, seq_length_out, embed_dim_head)
        query = query.unflatten(-1, [self.num_heads, self.head_dim]).transpose(-2, -3)
        # (batch_size, seq_length_in, embed_dim_all_heads)
        # -> (batch_size, seq_length_in, num_heads, embed_dim_head)
        # -> (batch_size, num_heads, seq_length_in, embed_dim_head)
        key = key.unflatten(-1, [self.num_heads, self.head_dim]).transpose(-2, -3)

        # (batch_size, seq_length_in, embed_dim_all_heads)
        # -> (batch_size, seq_length_in, num_heads, embed_dim_head)
        # -> (batch_size, num_heads, seq_length_in, embed_dim_head)
        value = value.unflatten(-1, [self.num_heads, self.head_dim]).transpose(-2, -3)

        # Step 3: Run scaled dot-product attention

        # TODO: If the q,k,v projections are all deterministic, this is quite inefficient,
        # since the output of each linear layer gets expanded to the number of samples.
        # Make sure in that case we only expand the output to the number of samples when needed.

        # (batch_size, num_heads, seq_length_out, embed_dim_head)
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=is_causal,
            scale=None,  # Defaults to 1/sqrt(embed_dim)
        )
        # (batch_size, num_heads, seq_length_out, embed_dim_head)
        # -> (batch_size, seq_length_out, num_heads, embed_dim_head)
        # -> (batch_size, seq_length_out, embed_dim_all_heads)
        attn_output = attn_output.transpose(-2, -3).flatten(-2)

        # Step 4. Apply output projection

        if self.out_proj is not None:
            # (batch_size, seq_length_out, embed_dim_all_heads)
            # -> (batch_size, seq_length_out, embed_dim_out)
            attn_output = self.out_proj(
                attn_output,
                sample_shape=sample_shape,
                generator=generator,
                input_contains_samples=True,
                parameter_samples=parameter_samples,
            )

        return attn_output

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ):

        # Ensure compatibility with nn.MultiheadAttention
        if prefix + "in_proj_weight" in state_dict:
            # Packed projection used for self-attention
            packed_qkv_proj_weight = state_dict.pop(prefix + "in_proj_weight")
            q_weight, k_weight, v_weight = torch.chunk(packed_qkv_proj_weight, 3, dim=0)
            state_dict[prefix + "q_proj.params.weight"] = q_weight
            state_dict[prefix + "k_proj.params.weight"] = k_weight
            state_dict[prefix + "v_proj.params.weight"] = v_weight

            if self.bias:
                packed_qkv_proj_bias = state_dict.pop(prefix + "in_proj_bias")
                q_bias, k_bias, v_bias = torch.chunk(packed_qkv_proj_bias, 3, dim=0)
                state_dict[prefix + "q_proj.params.bias"] = q_bias
                state_dict[prefix + "k_proj.params.bias"] = k_bias
                state_dict[prefix + "v_proj.params.bias"] = v_bias

        if prefix + "q_proj_weight" in state_dict:
            if self.bias:
                raise NotImplementedError(
                    "Only self-attention currently supports bias, "
                    "since nn.MultiheadAttention implements biases differently otherwise."
                )
            for nn_proj, bnn_proj in [
                ("q_proj", "q_proj"),
                ("k_proj", "k_proj"),
                ("v_proj", "v_proj"),
            ]:
                if prefix + nn_proj + "_weight" in state_dict:
                    state_dict[prefix + bnn_proj + ".params.weight"] = state_dict.pop(
                        prefix + nn_proj + "_weight"
                    )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
