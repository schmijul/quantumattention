import torch
import torch.nn as nn

from src.quantum_embedding import OptimizedQuantumEmbedding
from src.quantum_attention import HybridMultiHeadAttention


class TransformerClassifier(nn.Module):
    """Simple transformer-based text classifier.

    Parameters
    ----------
    embedding_layer : nn.Module
        Either ``nn.Embedding`` or ``OptimizedQuantumEmbedding``.
    attention_layer : nn.Module
        Attention module compatible with ``nn.MultiheadAttention`` interface.
    embedding_dim : int
        Dimension of embeddings.
    num_classes : int
        Number of output classes.
    """

    def __init__(self, embedding_layer: nn.Module, attention_layer: nn.Module,
                 embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embedding = embedding_layer
        self.attention = attention_layer
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        emb = emb.permute(1, 0, 2)  # (seq_len, batch, embed_dim)
        attn_out, _ = self.attention(emb, emb, emb)
        attn_out = attn_out.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
        x = self.ffn(attn_out)
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


class ClassicalTransformer(TransformerClassifier):
    """Transformer classifier with classical components."""

    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int) -> None:
        embedding = nn.Embedding(vocab_size, embedding_dim)
        attention = nn.MultiheadAttention(embedding_dim, num_heads=1)
        super().__init__(embedding, attention, embedding_dim, num_classes)


class HybridTransformer(TransformerClassifier):
    """Transformer classifier using quantum embedding and attention."""

    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int,
                 n_qubits: int = 6, n_layers: int = 2, shots: int = 1000) -> None:
        embedding = OptimizedQuantumEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            shots=shots,
        )
        attention = HybridMultiHeadAttention(
            embedding_dim,
            num_heads=1,
            n_qubits=n_qubits,
            shots=shots,
        )
        super().__init__(embedding, attention, embedding_dim, num_classes)
