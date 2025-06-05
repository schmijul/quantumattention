"""
Minimal Hybrid Transformer Test
Tests the hybrid transformer with extremely small parameters for quick validation.
"""

import torch
import time
from src.hybrid_transformer import HybridTransformer
from src.classical_transformer import ClassicalTransformer

def test_minimal_hybrid():
    print("🧪 Testing minimal hybrid transformer...")
    print("=" * 50)
    
    # ULTRA minimal parameters
    vocab_size = 5       # Only 5 words
    embedding_dim = 4    # Tiny dimension  
    num_classes = 2      # Binary classification
    n_qubits = 3         # Minimal qubits
    n_layers = 1         # Single layer
    shots = 50           # Few shots
    
    print(f"📊 Parameters:")
    print(f"   vocab_size: {vocab_size}")
    print(f"   embedding_dim: {embedding_dim}")
    print(f"   n_qubits: {n_qubits}")
    print(f"   shots: {shots}")
    print()
    
    # Test classical first
    print("🔵 Testing Classical Transformer...")
    start_time = time.time()
    classical = ClassicalTransformer(vocab_size, embedding_dim, num_classes)
    x = torch.randint(0, vocab_size, (1, 2))  # 1 sample, 2 tokens
    print(f"   Input: {x}")
    classical_out = classical(x)
    classical_time = time.time() - start_time
    print(f"   ✅ Classical works! Output: {classical_out}")
    print(f"   ⏱️  Time: {classical_time:.3f}s")
    print()
    
    # Test hybrid
    print("🟡 Testing Hybrid Transformer...")
    start_time = time.time()
    hybrid = HybridTransformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        n_qubits=n_qubits,
        n_layers=n_layers,
        shots=shots
    )
    print("   🔧 Hybrid model created")
    
    print("   🚀 Running forward pass...")
    hybrid_out = hybrid(x)
    hybrid_time = time.time() - start_time
    print(f"   ✅ Hybrid works! Output: {hybrid_out}")
    print(f"   ⏱️  Time: {hybrid_time:.3f}s")
    print(f"   📈 Slowdown: {hybrid_time/classical_time:.1f}x")
    print()
    
    # Test gradient computation
    print("🔄 Testing gradient computation...")
    loss = hybrid_out.sum()
    loss.backward()
    print("   ✅ Gradients computed successfully!")
    
    print("🎉 All tests passed! Hybrid transformer is working.")
    return True

if __name__ == "__main__":
    test_minimal_hybrid()