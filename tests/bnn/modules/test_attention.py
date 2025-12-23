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
@pytest.mark.parametrize(
    "cov",
    [None, {"q": None, "k": None, "v": params.FactorizedCovariance(), "out": None}],
)
def test_generalizes_pytorch_multi_head_attention_layer(
    embed_dim,
    num_heads,
    kdim,
    vdim,
    is_self_attention,
    cov,
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
        cov=cov,
    )
    attn_inferno.load_state_dict(attn_torch.state_dict(), strict=False)

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
    out_seq_inferno = attn_inferno(query, key, value, sample_shape=None)

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


@pytest.mark.parametrize(
    "layer,is_self_attention",
    [
        (
            bnn.MultiheadAttention(
                embed_dim=32,
                num_heads=4,
                cov=params.FactorizedCovariance(),
            ),
            True,
        ),
        (
            bnn.MultiheadAttention(
                embed_dim=32,
                num_heads=4,
                bias=True,
                cov=params.LowRankCovariance(4),
            ),
            True,
        ),
        (
            bnn.MultiheadAttention(
                embed_dim=32,
                num_heads=4,
                kdim=4,
                vdim=12,
                cov=params.FactorizedCovariance(),
            ),
            False,
        ),
    ],
)
@pytest.mark.parametrize("sample_shape", [(), (1,), (2,), (3, 2)])
@pytest.mark.parametrize("batch_shape", [(), (1,), (3,)])
def test_shape(sample_shape, batch_shape, layer, is_self_attention):
    """Test whether the output shape is correct."""
    generator = torch.Generator().manual_seed(0)

    num_tokens = 128
    embed_dim = layer.embed_dim
    kdim = layer.kdim
    vdim = layer.vdim
    query = torch.randn(*batch_shape, num_tokens, embed_dim, generator=generator)
    if is_self_attention:
        key = query
        value = query
    else:
        key = torch.randn(
            *batch_shape,
            num_tokens,
            embed_dim if kdim is None else kdim,
            generator=generator,
        )
        value = torch.randn(
            *batch_shape,
            num_tokens,
            embed_dim if vdim is None else vdim,
            generator=generator,
        )
    output = layer(query, key, value, sample_shape=sample_shape, generator=generator)

    assert output.shape == (*sample_shape, *batch_shape, num_tokens, embed_dim)
