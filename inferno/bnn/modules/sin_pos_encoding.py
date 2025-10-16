from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class SinusoidalPositionalEncoding(nn.Module):
    r"""Sinusoidal Positional Encoding.

    This module adds fixed sinusoidal positional embeddings ([Vaswani et al., 2017; Sec. 3.5](https://arxiv.org/abs/1706.03762)) 
    to input embeddings to provide the model with information about the relative or absolute position of tokens in a sequence.

    The sinusoidal positional encoding uses sine and cosine functions of different frequencies:

    $$
    \begin{align*}
        \operatorname{PE}(\text{pos}, 2i)   &= \sin(\text{base}^{-\frac{2i}{\text{embed\_dim}}} \cdot \text{pos}) \\
        \operatorname{PE}(\text{pos}, 2i+1) &= \cos(\text{base}^{-\frac{2i}{\text{embed\_dim}}} \cdot \text{pos})
    \end{align*}
    $$

    where $\text{pos}$ is the position index, $0\leq i \leq \frac{\text{embed\_dim}{2}$ is the embedding dimension 
    index and $\text{embed\_dim}$ is the embedding dimensionality.

    The encoding is designed so that each dimension of the positional encoding corresponds
    to a sinusoid with wavelengths forming a geometric progression from 2π to base·2π.
    This allows the model to easily learn to attend by relative positions.

    Notes:

    - The positional encodings are added to the input embeddings, so both must have
        the same embedding dimension (``embed_dim``).
    - If the input sequence length exceeds ``max_seq_len``, the encoding will be truncated
        to match the input length, which may cause issues. Ensure ``max_seq_len`` ≥ expected
        sequence lengths.
    - The encoding is deterministic and does not require training.


    ```python
    import torch
    from inferno.bnn.modules import SinusoidalPositionalEncoding

    # Create positional encoding for 512-dim embeddings
    pos_enc = SinusoidalPositionalEncoding(embed_dim=512)

    # Apply to input embeddings
    batch_size, seq_len, embed_dim = 32, 100, 512
    input_embeddings = torch.randn(batch_size, seq_len, embed_dim)
    output = pos_enc(input_embeddings)
    print(output.shape)  # torch.Size([32, 100, 512])
    ```

    :param embed_dim: The embedding dimension. Should be even for proper sine/cosine pairing.
    :param max_seq_len: Maximum sequence length to precompute encodings for.
    :param base: Base of the angular frequency.
    :param dtype: Data type for the positional encodings.
    :param device: Device to place the positional encodings on.
    """

    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.base = base
        self.max_seq_len = max_seq_len

        positions = torch.arange(
            self.max_seq_len,
            dtype=dtype,
            device=device,
        ).unsqueeze(1)

        # Compute the angular frequencies for each dimension
        angular_frequencies = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=dtype, device=device)
            * (-math.log(self.base) / embed_dim)
        )

        positional_encoding = torch.zeros(
            1, self.max_seq_len, embed_dim, dtype=dtype, device=device
        )

        # Apply sine to even indices (0, 2, 4, ...)
        positional_encoding[0, :, 0::2] = torch.sin(positions * angular_frequencies)

        # Apply cosine to odd indices (1, 3, 5, ...)
        positional_encoding[0, :, 1::2] = torch.cos(positions * angular_frequencies)

        self.register_buffer("positional_encoding", positional_encoding)

    def forward(
        self, x: Float[Tensor, "batch token embed_dim"]
    ) -> Float[Tensor, "batch token embed_dim"]:
        """Add positional encoding to input embeddings.

        :param x: Input embeddings.
        :return: Input embeddings with added positional encoding of the same shape.
        """
        if self.max_seq_len < x.shape[1]:
            raise ValueError(
                f"Maximum sequence length of positional encoding ({self.max_seq_len}) "
                f"is less than input sequence length ({x.shape[1]})."
            )

        # Add positional encoding to input embeddings
        # Only use the positional encodings up to the sequence length
        return x + self.positional_encoding[:, : x.size(1)]
