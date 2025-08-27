import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .quantum_embedding import OptimizedQuantumEmbedding


class TransformerLM(nn.Module):
    """Minimal transformer-style language model.

    - Pluggable embedding module (classical or quantum-embedding)
    - Single self-attention block + FFN
    - Learned positional embeddings
    - Causal masking for autoregressive next-token prediction

    Shapes:
      x: (batch, seq_len) -> logits: (batch, seq_len, vocab_size)
    """

    def __init__(
        self,
        embedding_layer: nn.Module,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 256,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.embedding = embedding_layer
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=False)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        # Causal mask cached for speed
        self.register_buffer("_causal_mask", self._generate_causal_mask(max_seq_len), persistent=False)

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        mask = torch.full((size, size), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        return mask

    def _get_causal_mask(self, seq_len: int) -> torch.Tensor:
        if self._causal_mask.size(0) < seq_len:
            self._causal_mask = self._generate_causal_mask(seq_len).to(self._causal_mask.device)
        return self._causal_mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for each position.

        Args:
          x: token indices (batch, seq_len)
        Returns:
          logits: (batch, seq_len, vocab_size)
        """
        bsz, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        tok = self.embedding(x)  # (B, T, C)
        pos_ids = torch.arange(seq_len, device=x.device)
        pos = self.pos_embedding(pos_ids)[None, :, :]  # (1, T, C)
        h = tok + pos

        # MHA expects (T, B, C)
        h = h.permute(1, 0, 2)
        mask = self._get_causal_mask(seq_len)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        h = self.ln1(h + attn_out)
        ff = self.ffn(h)
        h = self.ln2(h + ff)

        h = h.permute(1, 0, 2)  # (B, T, C)
        logits = self.head(h)    # (B, T, V)
        return logits


class ClassicalTransformerLM(TransformerLM):
    """Language model with classical embedding and attention."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 256,
        num_heads: int = 1,
    ) -> None:
        embedding = nn.Embedding(vocab_size, embed_dim)
        super().__init__(embedding, vocab_size, embed_dim, max_seq_len, num_heads)


class QuantumEmbeddingTransformerLM(TransformerLM):
    """Language model using a quantum embedding with classical attention/FFN."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 256,
        num_heads: int = 1,
        n_qubits: int = 6,
        n_layers: int = 2,
        shots: int = 300,
    ) -> None:
        q_embedding = OptimizedQuantumEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            shots=shots,
        )
        super().__init__(q_embedding, vocab_size, embed_dim, max_seq_len, num_heads)

