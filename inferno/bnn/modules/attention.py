from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bnn_mixin import BNNMixin

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class MultiHeadAttention(BNNMixin, nn.Module):
    """Attention layer (with multiple heads).

    Multi-head (self-)attention layer with an optional attention mask, allowing a model to jointly attend
    to information from different representation subspaces.

    The module supports nested or padded tensors and is based on the following
    [implementation](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html).

    :param embed_dim_q: Dimensionality of the query embedding.
    :param embed_dim_k: Dimensionality of the key embedding.
    :param embed_dim_v: Dimensionality of the value embedding.
    :param embed_dim_all_heads: Total embedding dim of combined heads post input projection. Each head
            has dim embed_dim_all_heads // num_heads
    :param num_heads: Number of attention heads.
    :param dropout: Dropout probability; if greater than 0.0, dropout is applied.
    :param bias: Whether to add bias to input projection.
    """

    def __init__(
        self,
        embed_dim_q: int,
        embed_dim_k: int,
        embed_dim_v: int,
        embed_dim_all_heads: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self._qkv_same_embed_dim = (
            embed_dim_q == embed_dim_k and embed_dim_q == embed_dim_v
        )
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(
                embed_dim_q, embed_dim_all_heads * 3, bias=bias, **factory_kwargs
            )
        else:
            self.q_proj = nn.Linear(
                embed_dim_q, embed_dim_all_heads, bias=bias, **factory_kwargs
            )
            self.k_proj = nn.Linear(
                embed_dim_k, embed_dim_all_heads, bias=bias, **factory_kwargs
            )
            self.v_proj = nn.Linear(
                embed_dim_v, embed_dim_all_heads, bias=bias, **factory_kwargs
            )
        embed_dim_out = embed_dim_q
        self.out_proj = nn.Linear(
            embed_dim_all_heads, embed_dim_out, bias=bias, **factory_kwargs
        )
        if embed_dim_all_heads % num_heads != 0:
            raise ValueError("Embedding dimension is not divisible by num_heads.")

        self.embed_dim_head = embed_dim_all_heads // num_heads
        self.bias = bias

    def forward(
        self,
        query: Float[Tensor, "*sample batch query_head query_token embed_dim_q"],
        key: Float[Tensor, "*sample batch head token embed_dim_k"],
        value: Float[Tensor, "*sample batch head token embed_dim_v"],
        attn_mask: Float[Tensor, "batch query_token token"] | None = None,
        is_causal: bool = False,
    ) -> Float[Tensor, "*sample batch query_head token embed_dim_v"]:
        """Computes scaled dot product attention on query, key and value tensors, using an optional attention mask.

        :param query: Query tensor
        :param key: Key tensor
        :param value: Value tensor
        :param attn_mask: Attention mask; shape must be broadcastable to the shape of attention weights.
                Two types of masks are supported.
                A boolean mask where a value of True indicates that the element *should* take part in attention.
                A float mask of the same type as query, key, value that is added to the attention score.
        :param is_causal: If set to true, the attention masking is a lower triangular matrix when the mask is a
                square matrix. The attention masking has the form of the upper left causal bias due to the alignment
                (see :class:`~torch.nn.attention.bias.CausalBias`) when the mask is a non-square matrix.
                An error is thrown if both ``attn_mask`` and ``is_causal`` are set.
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, embed_dim_all_heads) -> (N, L_t, num_heads, embed_dim_head) -> (N, num_heads, L_t, embed_dim_head)
        query = query.unflatten(-1, [self.num_heads, self.embed_dim_head]).transpose(
            1, 2
        )
        # (N, L_s, embed_dim_all_heads) -> (N, L_s, num_heads, embed_dim_head) -> (N, num_heads, L_s, embed_dim_head)
        key = key.unflatten(-1, [self.num_heads, self.embed_dim_head]).transpose(1, 2)
        # (N, L_s, embed_dim_all_heads) -> (N, L_s, num_heads, embed_dim_head) -> (N, num_heads, L_s, embed_dim_head)
        value = value.unflatten(-1, [self.num_heads, self.embed_dim_head]).transpose(
            1, 2
        )

        # Step 3. Run SDPA
        # (N, num_heads, L_t, embed_dim_head)
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=is_causal,
        )
        # (N, num_heads, L_t, embed_dim_head) -> (N, L_t, num_heads, embed_dim_head) -> (N, L_t, embed_dim_all_heads)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, embed_dim_all_heads) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output
