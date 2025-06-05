import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.quantum_embedding import QuantumEmbedding, OptimizedQuantumEmbedding


def test_quantum_embedding_forward_shape():
    embedding = QuantumEmbedding(
        vocab_size=3,
        embedding_dim=2,
        n_qubits=2,
        n_layers=1,
        shots=1,
        device_name="default.qubit",
    ).double()
    token_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    output = embedding(token_ids)
    assert output.shape == (2, 2, 2)


def test_optimized_quantum_embedding_cache():
    embedding = OptimizedQuantumEmbedding(
        vocab_size=4,
        embedding_dim=2,
        n_qubits=2,
        n_layers=1,
        shots=1,
    )
    token_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    embedding.clear_cache()
    assert len(embedding._quantum_cache) == 0
    output = embedding(token_ids)
    assert output.shape == (2, 2, 2)
    # should cache unique tokens 0,1,2
    assert len(embedding._quantum_cache) == 3
    embedding.clear_cache()
    assert len(embedding._quantum_cache) == 0
