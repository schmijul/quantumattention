import os
import sys
import torch
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.quantum_attention import QuantumAttention, HybridMultiHeadAttention


def test_compute_overlap_identity():
    attn = QuantumAttention(embed_dim=2, n_qubits=2, shots=None, device_name="default.qubit").double()
    vec = torch.tensor([1.0, 0.0], dtype=torch.float64)
    overlap = attn._compute_overlap(vec, vec)
    assert torch.isclose(overlap, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)


def test_quantum_attention_forward_shapes():
    attn = QuantumAttention(embed_dim=2, n_qubits=2, shots=None, device_name="default.qubit").double()
    q = torch.randn(3, 1, 2, dtype=torch.float64)
    k = torch.randn(3, 1, 2, dtype=torch.float64)
    v = torch.randn(3, 1, 2, dtype=torch.float64)
    out, weights = attn(q, k, v)
    assert out.shape == (3, 1, 2)
    assert weights.shape == (3, 1, 3)


def test_hybrid_multihead_attention_forward_shapes():
    mha = HybridMultiHeadAttention(embed_dim=2, n_qubits=2, shots=None, device_name="default.qubit").double()
    q = torch.randn(3, 1, 2, dtype=torch.float64)
    k = torch.randn(3, 1, 2, dtype=torch.float64)
    v = torch.randn(3, 1, 2, dtype=torch.float64)
    out, weights = mha(q, k, v)
    assert out.shape == (3, 1, 2)
    assert weights.shape == (3, 1, 3)


def test_hybrid_multihead_attention_multiple_heads_shapes():
    mha = HybridMultiHeadAttention(
        embed_dim=4,
        num_heads=2,
        n_qubits=2,
        shots=None,
        device_name="default.qubit",
    ).double()
    q = torch.randn(3, 1, 4, dtype=torch.float64)
    k = torch.randn(3, 1, 4, dtype=torch.float64)
    v = torch.randn(3, 1, 4, dtype=torch.float64)
    out, weights = mha(q, k, v)
    assert out.shape == (3, 1, 4)
    assert weights.shape == (3, 1, 3)


def test_hybrid_multihead_attention_invalid_head_split():
    with pytest.raises(ValueError):
        HybridMultiHeadAttention(
            embed_dim=5,
            num_heads=2,
            n_qubits=3,
            shots=None,
            device_name="default.qubit",
        )


def test_quantum_attention_respects_causal_mask():
    attn = QuantumAttention(
        embed_dim=2,
        n_qubits=2,
        shots=None,
        device_name="default.qubit",
        score_mode="fidelity",
    ).double()
    q = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    v = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 2.0]],
            [[3.0, 3.0]],
        ],
        dtype=torch.float64,
    )
    causal_mask = torch.tensor(
        [
            [False, True, True],
            [False, False, True],
            [False, False, False],
        ]
    )
    out, weights = attn(q, q, v, attn_mask=causal_mask)
    assert out.shape == (3, 1, 2)
    assert torch.isclose(weights[0, 0, 1], torch.tensor(0.0, dtype=torch.float64))
    assert torch.isclose(weights[0, 0, 2], torch.tensor(0.0, dtype=torch.float64))
    assert torch.isclose(weights[1, 0, 2], torch.tensor(0.0, dtype=torch.float64))


def test_quantum_attention_respects_key_padding_mask():
    attn = QuantumAttention(
        embed_dim=2,
        n_qubits=2,
        shots=None,
        device_name="default.qubit",
        score_mode="fidelity",
    ).double()
    q = torch.randn(3, 1, 2, dtype=torch.float64)
    k = torch.randn(3, 1, 2, dtype=torch.float64)
    v = torch.randn(3, 1, 2, dtype=torch.float64)
    key_padding_mask = torch.tensor([[False, False, True]])
    _, weights = attn(q, k, v, key_padding_mask=key_padding_mask)
    assert torch.allclose(weights[:, 0, 2], torch.zeros(3, dtype=torch.float64))
