import copy
import math

import numpy as np
import numpy.testing as npt
import torch
from torch import nn

from inferno import bnn
from inferno.bnn import params

import pytest


@pytest.mark.parametrize("embed_dim,num_heads", [(12, 4), (5, 1)])
@pytest.mark.parametrize(
    "kdim,vdim,is_self_attention",
    [(None, None, False), (None, None, True), (2, 3, False)],
)
def test_generalizes_pytorch_multi_head_attention_layer(
    embed_dim,
    num_heads,
    kdim,
    vdim,
    is_self_attention,
):
    """Test whether the BNN attention layer generalizes the PyTorch attention layer."""

    generator = torch.random.manual_seed(2134546)

    attn_torch = nn.MultiheadAttention(
        embed_dim,
        num_heads,
        kdim=kdim,
        vdim=vdim,
        bias=is_self_attention,
        batch_first=True,
    )

    attn_inferno = bnn.MultiheadAttention(
        embed_dim,
        num_heads,
        kdim=kdim,
        vdim=vdim,
        bias=is_self_attention,
    )
    attn_inferno.load_state_dict(attn_torch.state_dict())

    batch_size = 8
    num_tokens = 100

    query = torch.randn(batch_size, num_tokens, embed_dim, generator=generator)
    if is_self_attention:
        key = query
        value = query
    else:
        key = torch.randn(
            batch_size,
            num_tokens,
            embed_dim if kdim is None else kdim,
            generator=generator,
        )
        value = torch.randn(
            batch_size,
            num_tokens,
            embed_dim if vdim is None else vdim,
            generator=generator,
        )

    out_seq_torch = attn_torch(query, key, value, need_weights=False)[0]
    out_seq_inferno = attn_inferno(query, key, value)

    npt.assert_allclose(
        out_seq_torch.detach().numpy(),
        out_seq_inferno.detach().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("seed", [0, 45234, 42])
@pytest.mark.parametrize(
    "kdim,vdim,is_self_attention,is_causal",
    [
        (None, None, False, False),
        (None, None, True, False),
        (None, None, True, True),
        (2, 3, False, False),
    ],
)
def test_forward_is_deterministic_given_generator(
    seed, kdim, vdim, is_self_attention, is_causal
):
    """Test whether the forward method is deterministic given a generator."""
    generator = torch.Generator().manual_seed(262345)
    embed_dim = 16
    attn_layer = bnn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=4,
        kdim=kdim,
        vdim=vdim,
        cov=params.FactorizedCovariance(),
    )

    batch_size = 8
    num_tokens = 100

    query = torch.randn(batch_size, num_tokens, embed_dim, generator=generator)
    if is_self_attention:
        key = query
        value = query
    else:
        key = torch.randn(
            batch_size,
            num_tokens,
            embed_dim if kdim is None else kdim,
            generator=generator,
        )
        value = torch.randn(
            batch_size,
            num_tokens,
            embed_dim if vdim is None else vdim,
            generator=generator,
        )

    output1 = attn_layer(
        query,
        key,
        value,
        is_causal=is_causal,
        generator=torch.Generator().manual_seed(seed),
    )
    output2 = attn_layer(
        query,
        key,
        value,
        is_causal=is_causal,
        generator=torch.Generator().manual_seed(seed),
    )

    npt.assert_allclose(output1.detach().numpy(), output2.detach().numpy())


# @pytest.mark.parametrize("sample_shape", [(), (1,), (2,), (3, 2)])
# @pytest.mark.parametrize("batch_shape", [(), (1,), (3,)])
# @pytest.mark.parametrize(
#     "layer",
#     [
#         bnn.Linear(5, 3, cov=params.FactorizedCovariance()),
#         bnn.Linear(3, 2, bias=False, cov=params.LowRankCovariance(2)),
#         bnn.Linear(3, 2, bias=False, cov=None),
#     ],
# )
# def test_shape(sample_shape, batch_shape, layer):
#     """Test whether the output shape is correct."""
#     generator = torch.Generator().manual_seed(0)
#     input = torch.randn(*batch_shape, linear_layer.in_features, generator=generator)
#     output = linear_layer(input, sample_shape=sample_shape, generator=generator)

#     assert output.shape == (*sample_shape, *batch_shape, linear_layer.out_features)
