import torch.nn as nn

from .transformer_classifier import TransformerClassifier
from .quantum_embedding import OptimizedQuantumEmbedding
from .quantum_attention import HybridMultiHeadAttention


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
