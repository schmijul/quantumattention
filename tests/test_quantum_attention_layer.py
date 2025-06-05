import os
import sys
import torch

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
