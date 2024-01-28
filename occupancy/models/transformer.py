__all__ = ["transformer", "conditional_transformer"]

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch.utils.checkpoint import checkpoint
import torch._dynamo


@torch.jit.script
def fused_rmsnorm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    x = x * torch.rsqrt((x**2).mean(dim=-1, keepdim=True) + eps) * weight + bias
    return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=torch.bfloat16))

    def forward(self, x: Tensor) -> Tensor:
        return fused_rmsnorm(x, self.weight, self.bias, self.eps)  # eps of bfloat16


@torch.jit.script
def fused_swiglu(
    x: Tensor, w1: Tensor, b1: Optional[Tensor], w2: Tensor, b2: Optional[Tensor], w3: Tensor, b3: Optional[Tensor]
) -> Tensor:
    x1 = F.linear(x, w1, b1)
    x2 = F.linear(x, w2, b2)
    hidden = F.silu(x1) * x2
    return F.linear(hidden, w3, b3)


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features

        self.w1 = nn.Linear(
            in_features,
            hidden_features,
            dtype=torch.bfloat16,
        )
        self.w2 = nn.Linear(
            in_features,
            hidden_features,
            dtype=torch.bfloat16,
        )
        self.w3 = nn.Linear(
            hidden_features,
            out_features,
            dtype=torch.bfloat16,
        )
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.w1.weight, std=1 / math.sqrt(hidden_features))
        nn.init.normal_(self.w2.weight, std=1 / math.sqrt(hidden_features))
        nn.init.normal_(self.w3.weight, std=1 / math.sqrt(out_features))

        nn.init.constant_(self.w1.bias, 0)
        nn.init.constant_(self.w2.bias, 0)
        nn.init.constant_(self.w3.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return fused_swiglu(x, self.w1.weight, self.w1.bias, self.w2.weight, self.w2.bias, self.w3.weight, self.w3.bias)


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # NOTE: This could probably be moved to Triton

    # Handle a possible sequence length mismatch in between q and k
    cos = cos[..., : x.shape[-2], :]
    sin = sin[..., : x.shape[-2], :]

    return (x * cos.type_as(x)) + (rotate_half(x) * sin.type_as(x))


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim_model: int):
        super().__init__()
        self.dim_model = dim_model
        # Generate and save the inverse frequency buffer (non trainable)
        max_seq_length = 65536
        inv_freq = 1.0 / (
            max_seq_length ** (torch.arange(0, dim_model, 2, dtype=torch.float32, requires_grad=False) / dim_model)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        _cos_cached, s_sin_cached = self._update_cos_sin_tables(max_seq_length)
        self.register_buffer("_cos_cached", _cos_cached, persistent=False)
        self.register_buffer("_sin_cached", s_sin_cached, persistent=False)

    def _update_cos_sin_tables(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, dtype=torch.bfloat16, requires_grad=False)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        _cos_cached = emb.cos()[None, None, :, :]
        _sin_cached = emb.sin()[None, None, :, :]

        return _cos_cached, _sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


def _naive_scaled_dot_product_flash_attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
    return attn_weight @ V


class Attention(nn.Module):
    def __init__(self, hidden_states: int, num_heads: int, head_size: int):
        super().__init__()
        self.hidden_states = hidden_states
        self.num_heads = num_heads
        self.head_size = head_size

        self.q_proj = nn.Linear(
            hidden_states,
            num_heads * head_size,
            dtype=torch.bfloat16,
        )
        self.k_proj = nn.Linear(
            hidden_states,
            num_heads * head_size,
            dtype=torch.bfloat16,
        )
        self.v_proj = nn.Linear(
            hidden_states,
            num_heads * head_size,
            dtype=torch.bfloat16,
        )
        self.out_proj = nn.Linear(
            num_heads * head_size,
            hidden_states,
            dtype=torch.bfloat16,
        )
        self.rotary = RotaryEmbedding(head_size)

        nn.init.normal_(self.q_proj.weight, std=1 / math.sqrt(num_heads * head_size))
        nn.init.normal_(self.k_proj.weight, std=1 / math.sqrt(num_heads * head_size))
        nn.init.normal_(self.v_proj.weight, std=1 / math.sqrt(num_heads * head_size))
        nn.init.normal_(self.out_proj.weight, std=1 / math.sqrt(hidden_states))

        nn.init.constant_(self.q_proj.bias, 0)
        nn.init.constant_(self.k_proj.bias, 0)
        nn.init.constant_(self.v_proj.bias, 0)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, input_embeds: Tensor, attention_mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
        q = self.q_proj(input_embeds)
        k = self.k_proj(input_embeds)
        v = self.v_proj(input_embeds)

        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_size).transpose(-2, -3)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_size).transpose(-2, -3)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_size).transpose(-2, -3)
        q, k = self.rotary(q, k)

        # attn_weights = _naive_scaled_dot_product_flash_attention(q, k, v) for export
        attn_weights = F.scaled_dot_product_attention(q, k, v, attention_mask, is_causal=is_causal)

        attn_weights = attn_weights.transpose(-2, -3).flatten(-2)

        out = self.out_proj(attn_weights)

        return out


class CrossAttention(nn.Module):
    def __init__(self, hidden_states: int, num_heads: int, head_size: int):
        super().__init__()
        self.hidden_states = hidden_states
        self.num_heads = num_heads
        self.head_size = head_size

        self.q_proj = nn.Linear(
            hidden_states,
            num_heads * head_size,
            dtype=torch.bfloat16,
        )
        self.k_proj = nn.Linear(
            hidden_states,
            num_heads * head_size,
            dtype=torch.bfloat16,
        )
        self.v_proj = nn.Linear(
            hidden_states,
            num_heads * head_size,
            dtype=torch.bfloat16,
        )
        self.out_proj = nn.Linear(
            num_heads * head_size,
            hidden_states,
            dtype=torch.bfloat16,
        )
        self.rotary = RotaryEmbedding(head_size)

    def forward(
        self,
        input_embeds: Tensor,
        condition_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self.q_proj(input_embeds)
        k = self.k_proj(condition_embeds)
        v = self.v_proj(condition_embeds)

        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_size).transpose(-2, -3)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_size).transpose(-2, -3)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_size).transpose(-2, -3)
        q, k = self.rotary(q, k)

        attn_weights = F.scaled_dot_product_attention(q, k, v, attention_mask)

        attn_weights = attn_weights.transpose(-2, -3).flatten(-2)

        out = self.out_proj(attn_weights)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.ln1 = RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, head_size)

        self.ln2 = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, hidden_size * 8 // 3, hidden_size)

    def forward(self, input_embeds: Tensor, attention_mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
        residual = input_embeds
        input_embeds = self.ln1(input_embeds)
        input_embeds = self.self_attn(input_embeds, attention_mask, is_causal)
        input_embeds = input_embeds + residual

        residual = input_embeds
        input_embeds = self.ln2(input_embeds)
        input_embeds = self.mlp(input_embeds)
        input_embeds = input_embeds + residual

        return input_embeds


class ConditionalDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.ln1 = RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, head_size)

        self.ln2 = RMSNorm(hidden_size)
        self.cross_attn = CrossAttention(hidden_size, num_heads, head_size)

        self.ln3 = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, hidden_size * 8 // 3, hidden_size)

    def forward(
        self,
        input_embeds: Tensor,
        condition_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        residual = input_embeds
        input_embeds = self.ln1(input_embeds)
        input_embeds = self.self_attn(input_embeds, attention_mask)
        input_embeds = input_embeds + residual

        residual = input_embeds
        input_embeds = self.ln2(input_embeds)
        input_embeds = self.cross_attn(input_embeds, condition_embeds, attention_mask)
        input_embeds = input_embeds + residual

        residual = input_embeds
        input_embeds = self.ln3(input_embeds)
        input_embeds = self.mlp(input_embeds)
        input_embeds = input_embeds + residual

        return input_embeds


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size

        self.layers = nn.ModuleList([DecoderLayer(hidden_size, num_heads, head_size) for _ in range(num_layers)])

    def forward(self, input_embeds: Tensor, attention_mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
        for layer in self.layers:
            input_embeds = layer(input_embeds, attention_mask, is_causal)

        return input_embeds


class ConditionalTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size

        self.layers = nn.ModuleList(
            [ConditionalDecoderLayer(hidden_size, num_heads, head_size) for _ in range(num_layers)]
        )

    def forward(
        self,
        input_embeds: Tensor,
        condition_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        for layer in self.layers:
            input_embeds = layer(input_embeds, condition_embeds, attention_mask)

        return input_embeds


class AveragePooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.pooling = nn.Conv1d(hidden_size, hidden_size, 2, 2)
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.nonlinear = nn.SiLU(True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pooling(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.norm(x)
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


def conditional_transformer(
    hidden_size: int = 128,
    num_layers: int = 6,
    num_heads: int = 8,
    head_size: int = 64,
) -> ConditionalTransformer:
    return ConditionalTransformer(hidden_size, num_layers, num_heads, head_size)


def transformer(
    hidden_size: int = 128,
    num_layers: int = 6,
    num_heads: int = 8,
    head_size: int = 64,
) -> Transformer:
    return Transformer(hidden_size, num_layers, num_heads, head_size)
