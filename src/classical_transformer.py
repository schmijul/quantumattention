import torch.nn as nn

from .transformer_classifier import TransformerClassifier


class ClassicalTransformer(TransformerClassifier):
    """Transformer classifier with classical components."""

    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int) -> None:
        embedding = nn.Embedding(vocab_size, embedding_dim)
        attention = nn.MultiheadAttention(embedding_dim, num_heads=1)
        super().__init__(embedding, attention, embedding_dim, num_classes)
