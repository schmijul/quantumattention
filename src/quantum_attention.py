# quantum_attention.py

"""Quantum attention mechanism using PennyLane.

This module provides a single-head quantum attention layer implemented via
SWAP test and a wrapper class that mimics ``nn.MultiheadAttention`` for
compatibility with classical transformer code.
"""

from typing import Tuple

import torch
import torch.nn as nn
import pennylane as qml


class QuantumAttention(nn.Module):
    """Single-head quantum attention based on the SWAP test.

    Parameters
    ----------
    embed_dim: int
        Dimension of query, key and value vectors.
    n_qubits: int, optional
        Number of qubits used to encode vectors. Default is 6.
    shots: int, optional
        Number of measurement shots for the quantum device. Default is 1000.
    device_name: str, optional
        PennyLane device backend. ``"lightning.qubit"`` is used for speed.
    """

    def __init__(self, embed_dim: int, n_qubits: int = 6, shots: int = 1000,
                 device_name: str = "lightning.qubit") -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.shots = shots
        self.state_dim = 2 ** n_qubits
        if embed_dim > self.state_dim:
            raise ValueError("embed_dim must be <= 2**n_qubits")

        self.qdev = qml.device(device_name, wires=2 * n_qubits + 1, shots=shots)
        self.swap_test = self._create_swap_test()

    def _create_swap_test(self):
        """Create a PennyLane QNode performing the SWAP test."""

        @qml.qnode(self.qdev, interface="torch", diff_method="parameter-shift")
        def circuit(qvec, kvec):
            qml.AmplitudeEmbedding(qvec, wires=range(1, self.n_qubits + 1),
                                   normalize=True)
            qml.AmplitudeEmbedding(kvec, wires=range(self.n_qubits + 1,
                                                     2 * self.n_qubits + 1),
                                   normalize=True)
            qml.Hadamard(wires=0)
            for i in range(self.n_qubits):
                qml.CSWAP(wires=[0, 1 + i, 1 + self.n_qubits + i])
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        return circuit

    def _compute_overlap(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute squared overlap between ``q`` and ``k`` using the SWAP test."""
        q_pad = torch.zeros(self.state_dim, dtype=q.dtype, device=q.device)
        k_pad = torch.zeros(self.state_dim, dtype=k.dtype, device=k.device)

        q_norm = q / (torch.norm(q) + 1e-8)
        k_norm = k / (torch.norm(k) + 1e-8)

        q_pad[: self.embed_dim] = q_norm
        k_pad[: self.embed_dim] = k_norm

        z = self.swap_test(q_pad, k_pad)
        overlap = 0.5 * (1 + z)
        return overlap

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute quantum attention.

        Parameters
        ----------
        query, key, value: torch.Tensor
            Tensors of shape ``(seq_len, batch_size, embed_dim)``.

        Returns
        -------
        attn_output: torch.Tensor
            Tensor of shape ``(seq_len, batch_size, embed_dim)`` containing the
            attention outputs.
        attn_weights: torch.Tensor
            Attention probability weights of shape ``(seq_len, batch_size,
            seq_len)``.
        """
        seq_len, batch_size, _ = query.shape
        outputs = []
        weights_all = []

        for b in range(batch_size):
            q_vectors = query[:, b, :]
            k_vectors = key[:, b, :]
            v_vectors = value[:, b, :]
            attn_matrix = torch.zeros(seq_len, seq_len, dtype=query.dtype,
                                      device=query.device)
            for i in range(seq_len):
                for j in range(seq_len):
                    attn_matrix[i, j] = self._compute_overlap(q_vectors[i],
                                                              k_vectors[j])
            attn_probs = torch.softmax(attn_matrix, dim=-1)
            context = torch.matmul(attn_probs, v_vectors)
            outputs.append(context)
            weights_all.append(attn_probs)

        attn_output = torch.stack(outputs, dim=1)
        attn_weights = torch.stack(weights_all, dim=1)
        return attn_output, attn_weights


class HybridMultiHeadAttention(nn.Module):
    """Wrapper module that mirrors ``nn.MultiheadAttention`` using a single
    quantum head.

    Parameters
    ----------
    embed_dim: int
        Size of the input and output feature dimension.
    num_heads: int, optional
        Number of attention heads. Only ``1`` is currently supported.
    n_qubits: int, optional
        Number of qubits for the quantum attention head.
    shots: int, optional
        Number of measurement shots for the quantum device.
    device_name: str, optional
        PennyLane device backend name.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, n_qubits: int = 6,
                 shots: int = 1000, device_name: str = "lightning.qubit") -> None:
        super().__init__()
        if num_heads != 1:
            raise NotImplementedError("Only a single quantum head is supported")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.quantum_attn = QuantumAttention(embed_dim, n_qubits=n_qubits,
                                             shots=shots,
                                             device_name=device_name)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply quantum attention to ``query`` and ``key``/``value`` tensors."""
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        out, weights = self.quantum_attn(q, k, v)
        out = self.out_proj(out)
        return out, weights
