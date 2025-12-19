import numpy.testing as npt
import torch

from inferno import bnn

import pytest


@pytest.mark.parametrize(
    "batch_size, seq_len, embed_dim",
    [
        (1, 10, 16),
        (4, 20, 32),
        (8, 50, 64),
        (2, 5, 8),
        (3, 100, 128),
    ],
)
def test_same_size_as_input_after_positional_encoding(batch_size, seq_len, embed_dim):
    torch.manual_seed(14534)
    pos_enc = bnn.SinusoidalPositionalEncoding(embed_dim=embed_dim, max_seq_len=seq_len)
    x = torch.randn(batch_size, seq_len, embed_dim)
    out = pos_enc(x)
    assert out.shape == x.shape


def test_raises_valueerror_when_seq_len_exceeds_max_seq_len():
    embed_dim = 16
    max_seq_len = 10
    batch_size = 2
    seq_len = 12  # greater than max_seq_len
    pos_enc = bnn.SinusoidalPositionalEncoding(
        embed_dim=embed_dim, max_seq_len=max_seq_len
    )
    x = torch.randn(batch_size, seq_len, embed_dim)
    with pytest.raises(
        ValueError, match="Maximum sequence length of positional encoding"
    ):
        pos_enc(x)


@pytest.mark.parametrize(
    "batch_size, seq_len, embed_dim",
    [
        (1, 10, 16),
        (4, 20, 32),
        (3, 100, 128),
    ],
)
@pytest.mark.parametrize(
    "base",
    [1024, 4096, 10_000],
)
def test_implements_sinusoidal_positional_encoding_correctly(
    batch_size, seq_len, embed_dim, base
):
    torch.manual_seed(14534)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Inferno implementation of positional encoding
    pos_enc = bnn.SinusoidalPositionalEncoding(
        embed_dim=embed_dim, max_seq_len=seq_len, base=base
    )

    # Manual implementation of positional encoding
    positions = torch.arange(0, seq_len).reshape(-1, 1)
    angular_frequencies = (
        base ** (-torch.arange(0, embed_dim, 2) / embed_dim)
    ).reshape(1, -1)
    pos_enc_manual = torch.zeros(1, seq_len, embed_dim)
    pos_enc_manual[0, :, 0::2] = torch.sin(positions * angular_frequencies)
    pos_enc_manual[0, :, 1::2] = torch.cos(positions * angular_frequencies)
    pos_enc_manual = x + pos_enc_manual

    npt.assert_allclose(
        pos_enc(x).detach().numpy(),
        pos_enc_manual.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
