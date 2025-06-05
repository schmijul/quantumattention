import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.hybrid_transformer import ClassicalTransformer, HybridTransformer


def test_classical_transformer_forward_shape():
    model = ClassicalTransformer(vocab_size=5, embedding_dim=4, num_classes=3)
    tokens = torch.randint(0, 5, (2, 3), dtype=torch.long)
    logits = model(tokens)
    assert logits.shape == (2, 3)


def test_hybrid_transformer_forward_shape():
    model = HybridTransformer(
        vocab_size=5,
        embedding_dim=2,
        num_classes=2,
        n_qubits=2,
        n_layers=1,
        shots=1,
    )
    tokens = torch.randint(0, 5, (1, 2), dtype=torch.long)
    logits = model(tokens)
    assert logits.shape == (1, 2)
