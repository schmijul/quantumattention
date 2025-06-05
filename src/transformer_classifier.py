import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    """Simple transformer-based text classifier.

    Parameters
    ----------
    embedding_layer : nn.Module
        Embedding layer, either ``nn.Embedding`` or a quantum embedding.
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
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classifier."""
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        emb = emb.permute(1, 0, 2)  # (seq_len, batch, embed_dim)
        attn_out, _ = self.attention(emb, emb, emb)
        attn_out = attn_out.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
        x = self.ffn(attn_out)
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
