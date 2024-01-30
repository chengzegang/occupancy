__all__ = ["VectorQuantization"]

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def vector_quantize(input: Tensor, weight: Tensor, commitment_cost: float = 0.25) -> Tensor:
    # Reshape input to [batch_size, num_channels, *]
    input_shape = input.shape
    weight = weight.view(weight.shape[0], -1, weight.shape[-1])
    hidden = input.view(weight.shape[0], -1, weight.shape[-1])
    # Calculate distances
    distances = (
        torch.sum(hidden**2, dim=-1, keepdim=True)
        + torch.sum(weight**2, dim=-1)
        - 2 * torch.matmul(hidden, weight.transpose(-1, -2))
    )
    # Encoding
    encoding_indices = torch.argmin(distances, dim=-1, keepdim=True)
    encodings = torch.zeros(
        encoding_indices.shape[0],
        encoding_indices.shape[1],
        weight.shape[-2],
        device=hidden.device,
        dtype=hidden.dtype,
    )
    encodings.scatter_(-1, encoding_indices, 1)
    # Quantize and unflatten
    quantized = torch.matmul(encodings, weight).view(input_shape)
    # Loss
    e_latent_loss = F.mse_loss(quantized.detach(), input, reduction="none")
    q_latent_loss = F.mse_loss(quantized, input.detach(), reduction="none")
    loss = q_latent_loss + commitment_cost * e_latent_loss
    # Straight-through estimator
    quantized = input + (quantized - input).detach()
    return quantized, loss


class VectorQuantization(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, input: Tensor) -> Tensor:
        # Reshape input to [batch_size, num_channels, *]
        input_shape = input.shape
        hidden = input.view(-1, self.embedding_dim)
        # Calculate distances
        distances = (
            torch.sum(hidden**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(hidden, self._embedding.weight.t())
        )
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self.num_embeddings,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        encodings.scatter_(1, encoding_indices, 1)
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), input, reduction="none")
        q_latent_loss = F.mse_loss(quantized, input.detach(), reduction="none")
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # Straight-through estimator
        quantized = input + (quantized - input).detach()
        return quantized, loss
