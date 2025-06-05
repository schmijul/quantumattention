import time
import torch
import torch.nn as nn

from src.quantum_attention import HybridMultiHeadAttention


def main():
    embed_dim = 4
    seq_len = 3
    batch_size = 2

    torch.manual_seed(0)
    q = torch.randn(seq_len, batch_size, embed_dim)
    k = torch.randn(seq_len, batch_size, embed_dim)
    v = torch.randn(seq_len, batch_size, embed_dim)

    classical_mha = nn.MultiheadAttention(embed_dim, num_heads=1)
    classical_out, _ = classical_mha(q, k, v)
    print("Classical output shape:", classical_out.shape)

    hybrid_mha = HybridMultiHeadAttention(embed_dim, n_qubits=3, shots=1000)
    start = time.perf_counter()
    quantum_out, attn_weights = hybrid_mha(q, k, v)
    end = time.perf_counter()

    print("Quantum output shape:", quantum_out.shape)
    print("Attention weights shape:", attn_weights.shape)
    print(f"Quantum attention runtime: {end - start:.4f}s")


if __name__ == "__main__":
    main()
