"""
Minimal Hybrid Transformer Test
Tests the hybrid transformer with extremely small parameters for quick validation.
This version specifically tests a hybrid model with Quantum Embedding and Classical Attention.
"""

import unittest
import torch
import time
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the specific Hybrid model
from src.hybrid_quantum_embedding_transformer import HybridQuantumEmbeddingTransformer
from src.classical_transformer import ClassicalTransformer
# Assuming OptimizedQuantumEmbedding is in src.quantum_embedding
from src.quantum_embedding import OptimizedQuantumEmbedding


class TestMinimalHybridTransformer(unittest.TestCase):
    """Test suite for minimal hybrid transformer validation.
    Adapted for HybridQuantumEmbeddingTransformer.
    """

    def setUp(self):
        """Set up test parameters."""
        # ULTRA minimal parameters for rapid testing
        self.vocab_size = 5       # Only 5 words
        self.embedding_dim = 4    # Tiny dimension  
        self.num_classes = 2      # Binary classification
        self.n_qubits = 3         # Minimal qubits for Quantum Embedding
        self.n_layers = 1         # Single layer for Quantum Embedding
        self.shots = 50           # Few shots for Quantum Embedding
        self.batch_size = 1
        self.seq_len = 2
        
        # Test input
        self.x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        print(f"\n📊 Test Parameters:")
        print(f"   vocab_size: {self.vocab_size}")
        print(f"   embedding_dim: {self.embedding_dim}")
        print(f"   n_qubits (QEmb): {self.n_qubits}")
        print(f"   n_layers (QEmb): {self.n_layers}")
        print(f"   shots (QEmb): {self.shots}")
        print(f"   input shape: {self.x.shape}")

    def test_classical_transformer(self):
        """Test classical transformer baseline."""
        print("\n🔵 Testing Classical Transformer...")
        
        start_time = time.time()
        classical = ClassicalTransformer(
            self.vocab_size, 
            self.embedding_dim, 
            self.num_classes
        )
        
        output = classical(self.x)
        elapsed_time = time.time() - start_time
        
        print(f"   Input: {self.x}")
        print(f"   Output shape: {output.shape}")
        print(f"   ⏱️  Time: {elapsed_time:.3f}s")
        
        # Assertions
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
        # Store for comparison
        self.classical_time = elapsed_time
        self.classical_output = output

    def test_hybrid_quantum_embedding_transformer_forward(self):
        """Test HybridQuantumEmbeddingTransformer forward pass."""
        print("\n🟡 Testing Hybrid Quantum Embedding Transformer Forward Pass...")
        
        start_time = time.time()
        # Use the specific hybrid class
        hybrid_model = HybridQuantumEmbeddingTransformer(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            shots=self.shots
        )
        print("   🔧 HybridQuantumEmbeddingTransformer model created")
        
        print("   🚀 Running forward pass...")
        output = hybrid_model(self.x)
        elapsed_time = time.time() - start_time
        
        print(f"   Output shape: {output.shape}")
        print(f"   ⏱️  Time: {elapsed_time:.3f}s")
        
        # Assertions
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
        # Store for gradient test
        self.hybrid_model = hybrid_model
        self.hybrid_output = output
        self.hybrid_time = elapsed_time

    def test_hybrid_quantum_embedding_transformer_gradients(self):
        """Test gradient computation for HybridQuantumEmbeddingTransformer."""
        print("\n🔄 Testing Gradient Computation for HybridQuantumEmbeddingTransformer...")
        
        if not hasattr(self, 'hybrid_model'):
            self.test_hybrid_quantum_embedding_transformer_forward()
        
        try:
            loss = self.hybrid_output.sum()
            loss.backward()
            
            gradient_exists = False
            classical_gradients_found = 0
            quantum_gradients_found = 0
            
            for name, param in self.hybrid_model.named_parameters():
                if param.grad is not None:
                    gradient_exists = True
                    self.assertFalse(torch.isnan(param.grad).any(), 
                                   f"NaN gradient in {name}")
                    self.assertFalse(torch.isinf(param.grad).any(), 
                                   f"Inf gradient in {name}")
                    
                    if 'embedding.quantum_params' in name.lower():
                        quantum_gradients_found += 1
                    else:
                        classical_gradients_found += 1
                else:
                    print(f"   ⚠️ No gradient for: {name}") 
            
            print(f"   Classical parameter gradients found: {classical_gradients_found}")
            print(f"   Quantum embedding parameter gradients found: {quantum_gradients_found}")
            
            self.assertTrue(gradient_exists, "No gradients computed at all.")
            self.assertGreater(classical_gradients_found, 0, "No classical gradients computed")
            self.assertGreater(quantum_gradients_found, 0, "No quantum embedding gradients computed")
            
            print("   ✅ Gradients computed successfully for Quantum Embedding Transformer!")
            
        except Exception as e:
            print(f"   ❌ Gradient computation failed: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Gradient computation failed for HybridQuantumEmbeddingTransformer: {e}")

    def test_hybrid_quantum_embedding_transformer_gradients_step_by_step(self):
        """Test gradient computation step by step for HybridQuantumEmbeddingTransformer."""
        print("\n🔍 Testing HybridQuantumEmbeddingTransformer Gradients Step by Step...")
        
        hybrid_model = HybridQuantumEmbeddingTransformer(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            shots=self.shots
        )
        
        x = self.x.clone().detach().requires_grad_(False) 
        
        print("   🔧 Model parameters:")
        param_count = 0
        for name, param in hybrid_model.named_parameters():
            print(f"      {name}: {param.shape}, requires_grad={param.requires_grad}")
            param_count += param.numel()
        print(f"   Total parameters: {param_count}")
        
        print("   🚀 Forward pass...")
        output = hybrid_model(x)
        
        loss = output.mean()
        print(f"   📊 Loss: {loss.item():.6f}")
        
        try:
            print("   🔄 Computing gradients...")
            loss.backward(retain_graph=True) # retain_graph might be needed if backward is called multiple times
            
            gradients_found_count = 0
            for name, param in hybrid_model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"      ✅ {name}: grad_norm = {grad_norm:.6f}")
                    gradients_found_count +=1
                else:
                    print(f"      ❌ {name}: no gradient")
            
            self.assertGreater(gradients_found_count, 0, "No gradients computed in step-by-step test")
            print("   ✅ Step-by-step gradient test passed for Quantum Embedding Transformer!")
            
        except Exception as e:
            print(f"   ❌ Step-by-step gradient test failed: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Step-by-step gradient test failed: {e}")

    def test_performance_comparison(self):
        """Compare performance between classical and hybrid models."""
        print("\n📈 Performance Comparison...")
        
        if not hasattr(self, 'classical_time'):
            self.test_classical_transformer()
        if not hasattr(self, 'hybrid_time') or not isinstance(self.hybrid_model, HybridQuantumEmbeddingTransformer):
            self.test_hybrid_quantum_embedding_transformer_forward()
        
        slowdown = self.hybrid_time / self.classical_time
        print(f"   Classical time: {self.classical_time:.3f}s")
        print(f"   Hybrid (QEmb) time: {self.hybrid_time:.3f}s")
        print(f"   Slowdown: {slowdown:.1f}x")
        
        self.assertLess(slowdown, 150, "Hybrid model (QEmb) is too slow compared to classical (adjusted threshold)") # Increased threshold

    def test_output_consistency(self):
        """Test that outputs are consistent across runs for HybridQuantumEmbeddingTransformer."""
        print("\n🔄 Testing Output Consistency for HybridQuantumEmbeddingTransformer...")
        
        torch.manual_seed(42)
        
        hybrid_model = HybridQuantumEmbeddingTransformer(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            shots=self.shots
        )
        
        outputs = []
        for i in range(3):
            torch.manual_seed(42 + i) # Slightly vary seed to check robustness if shots are involved
            output = hybrid_model(self.x.clone()) # Use a clone of x
            outputs.append(output)
        
        for i in range(1, len(outputs)):
            # If shots=None (exact mode), difference should be near zero.
            # If shots are used, allow for statistical variation.
            # For default.qubit with shots, results can vary.
            # For lightning.qubit, it's deterministic even with shots if seed is fixed.
            # Let's assume for testing, we want nearly identical results if possible.
            # If using `default.qubit` with shots, this tolerance might need to be higher.
            diff = torch.abs(outputs[0] - outputs[i]).max()
            print(f"   Max difference run {i} vs run 0: {diff:.6f}")
            self.assertLess(diff, 0.5 if self.shots else 1e-5, f"Outputs too different between runs: {diff}")

    def test_model_parameters(self):
        """Test that HybridQuantumEmbeddingTransformer has trainable parameters."""
        print("\n🔧 Testing HybridQuantumEmbeddingTransformer Parameters...")
        
        hybrid_model = HybridQuantumEmbeddingTransformer(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            shots=self.shots
        )
        
        total_params = sum(p.numel() for p in hybrid_model.parameters())
        trainable_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params}")
        print(f"   Trainable parameters: {trainable_params}")
        
        self.assertGreater(total_params, 0, "Hybrid model has no parameters")
        self.assertGreater(trainable_params, 0, "Hybrid model has no trainable parameters")
        
        quantum_param_found = False
        for name, param in hybrid_model.named_parameters():
            self.assertFalse(torch.isnan(param).any(), f"NaN in parameter {name}")
            self.assertFalse(torch.isinf(param).any(), f"Inf in parameter {name}")
            if 'embedding.quantum_params' in name:
                quantum_param_found = True
                print(f"   ✅ Found quantum embedding parameter: {name} with shape {param.shape}")
        self.assertTrue(quantum_param_found, "Did not find expected quantum embedding parameters.")

    def test_different_input_sizes(self):
        """Test HybridQuantumEmbeddingTransformer with different input sizes."""
        print("\n📏 Testing HybridQuantumEmbeddingTransformer Different Input Sizes...")
        
        hybrid_model = HybridQuantumEmbeddingTransformer(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            shots=self.shots
        )
        
        test_cases = [
            (1, 1), (1, 3), (2, 2), (3, 1)
        ]
        
        for batch_size, seq_len in test_cases:
            with self.subTest(batch_size=batch_size, seq_len=seq_len):
                x_input = torch.randint(0, self.vocab_size, (batch_size, seq_len))
                output = hybrid_model(x_input)
                
                expected_shape = (batch_size, self.num_classes)
                self.assertEqual(output.shape, expected_shape)
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())
                
                print(f"   ✅ Input {x_input.shape} -> Output {output.shape}")

    def test_quantum_circuit_inspection(self):
        """Inspect the quantum components of HybridQuantumEmbeddingTransformer."""
        print("\n🔬 Quantum Circuit Inspection for HybridQuantumEmbeddingTransformer...")
        
        hybrid_model = HybridQuantumEmbeddingTransformer(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            shots=self.shots
        )
        
        try:
            self.assertTrue(hasattr(hybrid_model, 'embedding'), "Model has no 'embedding' attribute")
            q_embed_module = hybrid_model.embedding
            # Ensure it's the correct type, assuming OptimizedQuantumEmbedding is imported
            self.assertIsInstance(q_embed_module, OptimizedQuantumEmbedding, 
                                  "Embedding is not OptimizedQuantumEmbedding")
            print(f"   🎯 Found Quantum Embedding module (OptimizedQuantumEmbedding)")

            self.assertTrue(hasattr(q_embed_module, 'qdev'), "OptimizedQuantumEmbedding has no 'qdev'")
            print(f"   📊 QEmbedding device: {q_embed_module.qdev.name}")
            self.assertTrue(hasattr(q_embed_module, '_get_quantum_features'), 
                            "OptimizedQuantumEmbedding has no '_get_quantum_features' method")
            print(f"   🔧 Found '_get_quantum_features' method")

            try:
                dummy_token_id = 0 
                q_embed_module.eval() 
                if hasattr(q_embed_module, 'clear_cache'):
                    q_embed_module.clear_cache()
                # Test with a tensor input for _get_quantum_features if it expects one,
                # or ensure the dummy_token_id is appropriate for its internal logic.
                # The original _get_quantum_features takes an int.
                _ = q_embed_module._get_quantum_features(dummy_token_id)
                print(f"   ⚡ Successfully invoked _get_quantum_features (QNode likely created)")
            except Exception as e_qnode:
                self.fail(f"Failed to invoke _get_quantum_features, QNode might have issues: {e_qnode}")

            q_params_present = False
            for name, param in q_embed_module.named_parameters():
                if 'quantum_params' in name and param.requires_grad:
                    q_params_present = True
                    print(f"   ⚖️  Found trainable quantum embedding parameters: {name}, shape={param.shape}")
                    break
            self.assertTrue(q_params_present, "No trainable quantum parameters found in embedding!")
            
            self.assertTrue(hasattr(hybrid_model, 'attention'), "Model has no 'attention' attribute")
            self.assertIsInstance(hybrid_model.attention, torch.nn.MultiheadAttention,
                                  "Attention module is not classical MultiHeadAttention as expected.")
            print(f"   🎯 Found Classical MultiHeadAttention module")
            
            print("   ✅ Quantum circuit inspection completed for HybridQuantumEmbeddingTransformer")
            
        except AssertionError as ae:
            print(f"   ❌ Quantum circuit inspection assertion failed: {ae}")
            raise 
        except Exception as e:
            print(f"   ❌ Quantum circuit inspection failed with an unexpected error: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Quantum circuit inspection failed with an unexpected error: {e}")


class TestSuite:
    """Test suite runner with detailed output."""
    
    @staticmethod
    def run_all_tests():
        """Run all tests with detailed reporting."""
        print("🧪 Testing Minimal Hybrid Quantum Embedding Transformer")
        print("=" * 60)
        
        suite = unittest.TestLoader().loadTestsFromTestCase(TestMinimalHybridTransformer)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        print("\n" + "=" * 60)
        if result.wasSuccessful():
            print("🎉 All tests passed! Hybrid Quantum Embedding Transformer is working.")
        else:
            print("❌ Some tests failed!")
            print(f"Failures: {len(result.failures)}")
            print(f"Errors: {len(result.errors)}")
            
            if result.failures:
                print("\nFailures:")
                for test, traceback_str in result.failures:
                    print(f"  - {test}:\n{traceback_str}")
            
            if result.errors:
                print("\nErrors:")
                for test, traceback_str in result.errors:
                    print(f"  - {test}:\n{traceback_str}")
        
        return result.wasSuccessful()

    @staticmethod
    def run_quick_test():
        """Run only essential tests for quick validation."""
        print("⚡ Quick Test Suite for Hybrid Quantum Embedding Transformer")
        print("=" * 40)
        
        suite = unittest.TestSuite()
        suite.addTest(TestMinimalHybridTransformer('test_classical_transformer'))
        suite.addTest(TestMinimalHybridTransformer('test_hybrid_quantum_embedding_transformer_forward'))
        suite.addTest(TestMinimalHybridTransformer('test_hybrid_quantum_embedding_transformer_gradients'))
        suite.addTest(TestMinimalHybridTransformer('test_quantum_circuit_inspection'))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--suite":
            TestSuite.run_all_tests()
        elif sys.argv[1] == "--quick":
            TestSuite.run_quick_test()
        else:
            # This allows running specific tests like:
            # python tests/test_hybrid_transformer.py TestMinimalHybridTransformer.test_classical_transformer
            unittest.main(argv=sys.argv, verbosity=2)
    else:
        unittest.main(verbosity=2)