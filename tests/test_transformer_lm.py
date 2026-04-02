import torch
import torch.nn as nn

from src.transformer_lm import (
    ClassicalTransformerLM,
    QuantumAttentionTransformerLM,
    QuantumEmbeddingQuantumAttentionTransformerLM,
    QuantumEmbeddingTransformerLM,
)


def test_classical_lm_forward_backward():
    V = 64
    T = 8
    B = 2
    C = 8
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))
    model = ClassicalTransformerLM(vocab_size=V, embed_dim=C, max_seq_len=32)
    logits = model(x)
    assert logits.shape == (B, T, V)
    loss = nn.CrossEntropyLoss()(logits.reshape(B*T, V), y.reshape(B*T))
    loss.backward()


def test_quantum_embedding_lm_forward_backward():
    V = 64
    T = 6
    B = 2
    C = 6
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))
    model = QuantumEmbeddingTransformerLM(
        vocab_size=V,
        embed_dim=C,
        max_seq_len=32,
        n_qubits=5,
        n_layers=1,
        shots=50,
    )
    logits = model(x)
    assert logits.shape == (B, T, V)
    loss = nn.CrossEntropyLoss()(logits.reshape(B*T, V), y.reshape(B*T))
    loss.backward()


def test_quantum_attention_lm_forward_backward():
    V = 64
    T = 6
    B = 2
    C = 6
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))
    model = QuantumAttentionTransformerLM(
        vocab_size=V,
        embed_dim=C,
        max_seq_len=32,
        n_qubits=3,
        shots=None,
        score_mode="fidelity",
    )
    logits = model(x)
    assert logits.shape == (B, T, V)
    loss = nn.CrossEntropyLoss()(logits.reshape(B*T, V), y.reshape(B*T))
    loss.backward()


def test_full_quantum_lm_forward_backward():
    V = 64
    T = 5
    B = 2
    C = 4
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))
    model = QuantumEmbeddingQuantumAttentionTransformerLM(
        vocab_size=V,
        embed_dim=C,
        max_seq_len=32,
        n_qubits=3,
        n_layers=1,
        shots=20,
        score_mode="fidelity",
    )
    logits = model(x)
    assert logits.shape == (B, T, V)
    loss = nn.CrossEntropyLoss()(logits.reshape(B*T, V), y.reshape(B*T))
    loss.backward()
