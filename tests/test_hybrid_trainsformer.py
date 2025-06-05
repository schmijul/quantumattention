import torch
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.hybrid_transformer import HybridTransformer
from src.classical_transformer import ClassicalTransformer


def test_minimal_hybrid_execution():
    # Small configuration for quick execution
    vocab_size = 5
    embedding_dim = 4
    num_classes = 2
    n_qubits = 3
    n_layers = 1
    shots = 50

    x = torch.randint(0, vocab_size, (1, 2))

    classical = ClassicalTransformer(vocab_size, embedding_dim, num_classes)
    classical_out = classical(x)

    hybrid = HybridTransformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        n_qubits=n_qubits,
        n_layers=n_layers,
        shots=shots,
    )
    hybrid_out = hybrid(x)

    # both models should produce logits of the same shape
    assert classical_out.shape == hybrid_out.shape == (1, num_classes)

    # gradients should compute without error
    hybrid_out.sum().backward()
