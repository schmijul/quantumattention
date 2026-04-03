"""Classifier variants that use quantum attention."""

from __future__ import annotations

from typing import Dict

import torch.nn as nn

from .benchmarking import count_parameters
from .quantum_attention import HybridMultiHeadAttention
from .quantum_embedding import OptimizedQuantumEmbedding
from .transformer_classifier import TransformerClassifier


class QuantumAttentionTransformer(TransformerClassifier):
    """Classifier with classical embeddings and quantum attention."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        n_qubits: int = 3,
        shots: int = 300,
        num_heads: int = 1,
        device_name: str = "lightning.qubit",
        score_mode: str = "fidelity",
    ) -> None:
        embedding = nn.Embedding(vocab_size, embedding_dim)
        attention = HybridMultiHeadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            n_qubits=n_qubits,
            shots=shots,
            device_name=device_name,
            score_mode=score_mode,
        )
        super().__init__(embedding, attention, embedding_dim, num_classes)
        self.n_qubits = n_qubits
        self.shots = shots
        self.num_heads = num_heads
        self.score_mode = score_mode

    def get_quantum_info(self) -> Dict[str, object]:
        counts = count_parameters(self)
        return {
            **counts,
            "n_qubits": self.n_qubits,
            "shots": self.shots,
            "num_heads": self.num_heads,
            "score_mode": self.score_mode,
        }


class QuantumEmbeddingQuantumAttentionTransformer(TransformerClassifier):
    """Classifier with quantum embeddings and quantum attention."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        n_qubits: int = 3,
        n_layers: int = 2,
        shots: int = 300,
        num_heads: int = 1,
        device_name: str = "lightning.qubit",
        score_mode: str = "fidelity",
    ) -> None:
        embedding = OptimizedQuantumEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            shots=shots,
        )
        attention = HybridMultiHeadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            n_qubits=n_qubits,
            shots=shots,
            device_name=device_name,
            score_mode=score_mode,
        )
        super().__init__(embedding, attention, embedding_dim, num_classes)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.num_heads = num_heads
        self.score_mode = score_mode

    def get_quantum_info(self) -> Dict[str, object]:
        counts = count_parameters(self)
        quantum_params = sum(p.numel() for p in self.embedding.parameters())
        return {
            **counts,
            "quantum_parameters": quantum_params,
            "classical_parameters": counts["total_parameters"] - quantum_params,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "shots": self.shots,
            "num_heads": self.num_heads,
            "score_mode": self.score_mode,
        }
