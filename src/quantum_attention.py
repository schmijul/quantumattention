# quantum_attention.py

"""Quantum attention layers with transformer-style masking support."""

from typing import Optional, Tuple

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumAttention(nn.Module):
    """Single-head quantum attention.

    The module supports two scoring backends:
    - ``swap_test``: uses a PennyLane QNode for pairwise overlap estimation
    - ``fidelity``: computes the same normalized-state overlap directly in
      PyTorch, which is much faster for development and benchmarking
    """

    def __init__(
        self,
        embed_dim: int,
        n_qubits: int = 6,
        shots: int = 1000,
        device_name: str = "lightning.qubit",
        score_mode: str = "swap_test",
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.shots = shots
        self.score_mode = score_mode
        self.state_dim = 2 ** n_qubits
        if embed_dim > self.state_dim:
            raise ValueError("embed_dim must be <= 2**n_qubits")
        if score_mode not in {"swap_test", "fidelity"}:
            raise ValueError("score_mode must be 'swap_test' or 'fidelity'")

        self.qdev = None
        self.swap_test = None
        if score_mode == "swap_test":
            try:
                self.qdev = qml.device(device_name, wires=2 * n_qubits + 1, shots=shots)
            except Exception:
                self.qdev = qml.device("default.qubit", wires=2 * n_qubits + 1, shots=shots)
            self.swap_test = self._create_swap_test()

    def _create_swap_test(self):
        """Create a PennyLane QNode performing the SWAP test."""

        @qml.qnode(self.qdev, interface="torch", diff_method="parameter-shift")
        def circuit(qvec, kvec):
            qml.AmplitudeEmbedding(
                qvec,
                wires=range(1, self.n_qubits + 1),
                normalize=True,
            )
            qml.AmplitudeEmbedding(
                kvec,
                wires=range(self.n_qubits + 1, 2 * self.n_qubits + 1),
                normalize=True,
            )
            qml.Hadamard(wires=0)
            for i in range(self.n_qubits):
                qml.CSWAP(wires=[0, 1 + i, 1 + self.n_qubits + i])
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        return circuit

    def _prepare_states(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize and pad vectors for amplitude-style state comparisons."""
        norms = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
        normalized = x / norms
        pad_width = self.state_dim - self.embed_dim
        if pad_width > 0:
            normalized = F.pad(normalized, (0, pad_width))
        return normalized

    def _compute_overlap(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute normalized-state overlap between ``q`` and ``k``."""
        q_state = self._prepare_states(q.unsqueeze(0).unsqueeze(0))[0, 0]
        k_state = self._prepare_states(k.unsqueeze(0).unsqueeze(0))[0, 0]
        return self._compute_overlap_from_states(q_state, k_state)

    def _compute_overlap_from_states(
        self,
        q_state: torch.Tensor,
        k_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute overlap from preprocessed states."""
        if self.score_mode == "fidelity":
            return torch.dot(q_state, k_state).pow(2)

        z = self.swap_test(q_state, k_state)
        return 0.5 * (1 + z)

    def _compute_scores(
        self,
        q_states: torch.Tensor,
        k_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return a dense attention score matrix for one batch item."""
        if self.score_mode == "fidelity":
            return torch.matmul(q_states, k_states.transpose(0, 1)).pow(2)

        target_len = q_states.size(0)
        source_len = k_states.size(0)
        scores = torch.zeros(
            target_len,
            source_len,
            dtype=q_states.dtype,
            device=q_states.device,
        )
        for i in range(target_len):
            for j in range(source_len):
                scores[i, j] = self._compute_overlap_from_states(q_states[i], k_states[j])
        return scores

    def _expand_attn_mask(
        self,
        attn_mask: Optional[torch.Tensor],
        target_len: int,
        source_len: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attn_mask is None:
            return None

        mask = attn_mask.to(device=device)
        if mask.dim() == 2:
            if mask.shape != (target_len, source_len):
                raise ValueError("2D attn_mask must have shape (target_len, source_len)")
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        elif mask.dim() == 3:
            if mask.size(-2) != target_len or mask.size(-1) != source_len:
                raise ValueError("3D attn_mask must have shape (batch_or_head, target_len, source_len)")
            if mask.size(0) == 1:
                mask = mask.expand(batch_size, -1, -1)
            elif mask.size(0) != batch_size:
                raise ValueError("3D attn_mask first dimension must be 1 or batch_size")
        else:
            raise ValueError("attn_mask must be 2D or 3D")

        if mask.dtype == torch.bool:
            additive = torch.zeros(mask.shape, dtype=dtype, device=device)
            additive = additive.masked_fill(mask, float("-inf"))
            return additive

        return mask.to(dtype=dtype)

    def _expand_key_padding_mask(
        self,
        key_padding_mask: Optional[torch.Tensor],
        target_len: int,
        source_len: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if key_padding_mask is None:
            return None

        mask = key_padding_mask.to(device=device)
        if mask.dim() != 2 or mask.shape != (batch_size, source_len):
            raise ValueError("key_padding_mask must have shape (batch_size, source_len)")

        if mask.dtype == torch.bool:
            additive = torch.zeros(
                mask.size(0),
                target_len,
                source_len,
                dtype=dtype,
                device=device,
            )
            additive = additive.masked_fill(mask.unsqueeze(1), float("-inf"))
            return additive

        return mask.to(dtype=dtype).unsqueeze(1).expand(-1, target_len, -1)

    def _masked_softmax(
        self,
        scores: torch.Tensor,
        additive_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        masked_scores = scores if additive_mask is None else scores + additive_mask
        valid_rows = torch.isfinite(masked_scores).any(dim=-1, keepdim=True)
        safe_scores = torch.where(valid_rows, masked_scores, torch.zeros_like(masked_scores))
        probs = torch.softmax(safe_scores, dim=-1)
        return torch.where(valid_rows, probs, torch.zeros_like(probs))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute quantum attention for tensors of shape ``(T, B, C)``."""
        del average_attn_weights  # Single-head attention already returns one matrix.

        target_len, batch_size, _ = query.shape
        source_len = key.shape[0]

        q_states = self._prepare_states(query)
        k_states = self._prepare_states(key)

        attn_mask_expanded = self._expand_attn_mask(
            attn_mask,
            target_len=target_len,
            source_len=source_len,
            batch_size=batch_size,
            dtype=query.dtype,
            device=query.device,
        )
        key_padding_expanded = self._expand_key_padding_mask(
            key_padding_mask,
            target_len=target_len,
            source_len=source_len,
            batch_size=batch_size,
            dtype=query.dtype,
            device=query.device,
        )
        total_mask = None
        if attn_mask_expanded is not None and key_padding_expanded is not None:
            total_mask = attn_mask_expanded + key_padding_expanded
        elif attn_mask_expanded is not None:
            total_mask = attn_mask_expanded
        elif key_padding_expanded is not None:
            total_mask = key_padding_expanded

        outputs = []
        weights_all = []
        for batch_idx in range(batch_size):
            score_matrix = self._compute_scores(q_states[:, batch_idx, :], k_states[:, batch_idx, :])
            batch_mask = None if total_mask is None else total_mask[batch_idx]
            attn_probs = self._masked_softmax(score_matrix, batch_mask)
            context = torch.matmul(attn_probs, value[:, batch_idx, :])
            outputs.append(context)
            if need_weights:
                weights_all.append(attn_probs)

        attn_output = torch.stack(outputs, dim=1)
        if not need_weights:
            return attn_output, None

        attn_weights = torch.stack(weights_all, dim=1)
        return attn_output, attn_weights


class HybridMultiHeadAttention(nn.Module):
    """Multi-head wrapper with quantum attention heads."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        n_qubits: int = 6,
        shots: int = 1000,
        device_name: str = "lightning.qubit",
        score_mode: str = "swap_test",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.quantum_attn_heads = nn.ModuleList(
            [
                QuantumAttention(
                    self.head_dim,
                    n_qubits=n_qubits,
                    shots=shots,
                    device_name=device_name,
                    score_mode=score_mode,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply quantum attention to ``query``/``key``/``value`` tensors."""
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        seq_len, batch_size, _ = q.shape
        q = q.reshape(seq_len, batch_size, self.num_heads, self.head_dim)
        k = k.reshape(seq_len, batch_size, self.num_heads, self.head_dim)
        v = v.reshape(seq_len, batch_size, self.num_heads, self.head_dim)

        head_outputs = []
        head_weights = []
        for head_idx, head_attn in enumerate(self.quantum_attn_heads):
            out_h, w_h = head_attn(
                q[:, :, head_idx, :],
                k[:, :, head_idx, :],
                v[:, :, head_idx, :],
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
            )
            head_outputs.append(out_h)
            if need_weights:
                head_weights.append(w_h)

        out = torch.stack(head_outputs, dim=2).reshape(seq_len, batch_size, self.embed_dim)
        out = self.out_proj(out)

        if not need_weights:
            return out, None

        weights = torch.stack(head_weights, dim=0)
        if average_attn_weights:
            return out, weights.mean(dim=0)
        return out, weights.permute(1, 2, 0, 3)
