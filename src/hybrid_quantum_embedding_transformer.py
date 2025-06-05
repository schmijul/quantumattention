# src/hybrid_quantum_embedding_transformer.py


"""
Comparison: Classical Transformer vs. Hybrid Transformer with Quantum Embedding

This script trains and compares two transformer models for text classification:
1.  A purely classical transformer.
2.  A hybrid transformer that utilizes a Quantum Embedding Layer, while the
    rest of its architecture (attention, feed-forward) remains classical.

**Objective:**
The primary goal is to evaluate the performance (loss and accuracy) of a
transformer model where the standard classical embedding layer is replaced
by a Quantum Embedding Layer.

**Data Used:**
- A synthetic sentiment dataset is generated for this comparison.
- This dataset consists of token sequences where the sentiment (binary: positive/negative)
  is determined by the prevalence of high-value tokens (e.g., 10-19 for positive)
  versus low-value tokens (e.g., 1-9 for negative) within the sequence.
- This approach avoids external dependencies and provides a controlled
  testing environment.
- Key data parameters (vocabulary size, sequence length, train/validation split sizes)
  are configurable at the beginning of the script.

**Models Compared:**
1.  `ClassicalTransformer`:
    - A standard transformer architecture using a classical embedding
      (`nn.Embedding`) and classical Multi-Head Attention (`nn.MultiheadAttention`).
2.  `HybridQuantumEmbeddingTransformer`:
    - A hybrid architecture incorporating:
        - `OptimizedQuantumEmbedding`: Replaces the classical `nn.Embedding`.
          It employs Parameterized Quantum Circuits (PQCs) where each token's
          unique quantum parameters are learned. Measurements from these circuits
          yield classical features for the embedding.
        - `nn.MultiheadAttention`: A standard classical attention mechanism.
    - The remaining transformer blocks (feed-forward networks, layer normalization)
      are also classical.

**Training Process:**
- Both models are trained on the same synthetic dataset for a specified
  number of epochs.
- Training and validation loss, along with accuracy, are logged for each epoch.
- Hyperparameters for both classical and quantum components (e.g., `EMBED_DIM`,
  `N_QUBITS`, `N_LAYERS`, `SHOTS`) are configurable at the top of the script.

**Output:**
- Console output detailing the training progress and final performance metrics.
- A plot (`quantum_embedding_comparison.png`) visualizing the training/validation
  loss and accuracy curves for both models.

"""

import torch.nn as nn
from .transformer_classifier import TransformerClassifier
from .quantum_embedding import OptimizedQuantumEmbedding


class HybridQuantumEmbeddingTransformer(TransformerClassifier):
    """
    Hybrid Transformer using:
    - Quantum Embedding Layer (trainable quantum parameters)
    - Classical Multi-Head Attention (proven, stable)
    - Classical components for the rest
    """

    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int, 
        num_classes: int,
        n_qubits: int = 6, 
        n_layers: int = 2, 
        shots: int = 1000,
        num_heads: int = 1
    ) -> None:
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            num_classes: Number of output classes
            n_qubits: Number of qubits for quantum embedding
            n_layers: Number of quantum circuit layers
            shots: Number of quantum measurement shots
            num_heads: Number of attention heads (classical)
        """
        
        # ✅ QUANTUM EMBEDDING with trainable parameters
        embedding = OptimizedQuantumEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            shots=shots,
        )
        
        # ✅ CLASSICAL ATTENTION (proven and stable)
        attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=False  # (seq_len, batch, embed_dim)
        )
        
        # Initialize parent class
        super().__init__(embedding, attention, embedding_dim, num_classes)
        
        # Store quantum parameters for debugging
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots

    def get_quantum_info(self):
        """Get information about quantum components"""
        total_params = sum(p.numel() for p in self.parameters())
        quantum_params = sum(p.numel() for p in self.embedding.parameters())
        classical_params = total_params - quantum_params
        
        return {
            'total_parameters': total_params,
            'quantum_parameters': quantum_params,
            'classical_parameters': classical_params,
            'quantum_ratio': quantum_params / total_params,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'shots': self.shots
        }