from typing import Optional

import torch
import torch.nn as nn

from .quantum_attention import HybridMultiHeadAttention
from .quantum_embedding import OptimizedQuantumEmbedding


class TransformerLM(nn.Module):
    """Minimal transformer-style language model with pluggable attention."""

    def __init__(
        self,
        embedding_layer: nn.Module,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 256,
        num_heads: int = 1,
        attention_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.embedding = embedding_layer
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.attn = attention_layer or nn.MultiheadAttention(
            embed_dim,
            num_heads=num_heads,
            batch_first=False,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        self.register_buffer(
            "_causal_mask",
            self._generate_causal_mask(max_seq_len),
            persistent=False,
        )

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        mask = torch.full((size, size), float("-inf"))
        return torch.triu(mask, diagonal=1)

    def _get_causal_mask(self, seq_len: int) -> torch.Tensor:
        if self._causal_mask.size(0) < seq_len:
            self._causal_mask = self._generate_causal_mask(seq_len).to(self._causal_mask.device)
        return self._causal_mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(batch, seq_len, vocab_size)``."""
        _, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        tok = self.embedding(x)
        pos_ids = torch.arange(seq_len, device=x.device)
        pos = self.pos_embedding(pos_ids)[None, :, :]
        h = tok + pos

        h = h.permute(1, 0, 2)
        mask = self._get_causal_mask(seq_len).to(dtype=h.dtype, device=h.device)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=mask,
            need_weights=False,
        )
        h = self.ln1(h + attn_out)
        ff = self.ffn(h)
        h = self.ln2(h + ff)

        h = h.permute(1, 0, 2)
        return self.head(h)


class ClassicalTransformerLM(TransformerLM):
    """Language model with classical embedding and classical attention."""

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
    """Language model using a quantum embedding with classical attention."""

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


class QuantumAttentionTransformerLM(TransformerLM):
    """Language model using classical embeddings with quantum attention."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 256,
        num_heads: int = 1,
        n_qubits: int = 6,
        shots: int = 300,
        device_name: str = "lightning.qubit",
        score_mode: str = "fidelity",
    ) -> None:
        embedding = nn.Embedding(vocab_size, embed_dim)
        attention = HybridMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            n_qubits=n_qubits,
            shots=shots,
            device_name=device_name,
            score_mode=score_mode,
        )
        super().__init__(
            embedding,
            vocab_size,
            embed_dim,
            max_seq_len,
            num_heads,
            attention_layer=attention,
        )


class QuantumEmbeddingQuantumAttentionTransformerLM(TransformerLM):
    """Language model using quantum embeddings and quantum attention."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 256,
        num_heads: int = 1,
        n_qubits: int = 6,
        n_layers: int = 2,
        shots: int = 300,
        device_name: str = "lightning.qubit",
        score_mode: str = "fidelity",
    ) -> None:
        q_embedding = OptimizedQuantumEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            shots=shots,
        )
        attention = HybridMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            n_qubits=n_qubits,
            shots=shots,
            device_name=device_name,
            score_mode=score_mode,
        )
        super().__init__(
            q_embedding,
            vocab_size,
            embed_dim,
            max_seq_len,
            num_heads,
            attention_layer=attention,
        )
