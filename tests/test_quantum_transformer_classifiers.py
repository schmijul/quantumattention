import torch
import torch.nn as nn

from src.quantum_transformer_classifiers import (
    QuantumAttentionTransformer,
    QuantumEmbeddingQuantumAttentionTransformer,
)


def test_quantum_attention_classifier_forward_backward():
    V = 32
    T = 6
    B = 2
    C = 8
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, 2, (B,))
    model = QuantumAttentionTransformer(
        vocab_size=V,
        embedding_dim=C,
        num_classes=2,
        n_qubits=3,
        shots=None,
        score_mode="fidelity",
    )
    logits = model(x)
    assert logits.shape == (B, 2)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()


def test_full_quantum_classifier_forward_backward():
    V = 32
    T = 6
    B = 2
    C = 8
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, 2, (B,))
    model = QuantumEmbeddingQuantumAttentionTransformer(
        vocab_size=V,
        embedding_dim=C,
        num_classes=2,
        n_qubits=3,
        n_layers=1,
        shots=20,
        score_mode="fidelity",
    )
    logits = model(x)
    assert logits.shape == (B, 2)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
