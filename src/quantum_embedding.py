# quantum_embedding.py


"""
Quantum Embedding Layer Logic

CONCEPT:
- Replaces classical lookup table with parameterized quantum circuits (PQC)
- Each token gets unique quantum parameters that define a quantum state
- Quantum measurements extract features for classical post-processing

PROCESS:
1. Token ID → Quantum Parameters (trainable, per-token)
2. Parameters → Quantum Circuit → Quantum State |ψ⟩
3. Pauli-Z measurements → Classical feature vector
4. Linear layer → Final embedding vector

QUANTUM CIRCUIT:
- n_qubits qubits initialized to |0⟩^n
- n_layers of RY/RZ rotations + CNOT entangling gates
- Expectation values ⟨Z_i⟩ for each qubit as features

ADVANTAGES:
- Exponential Hilbert space (2^n_qubits dimensions)
- Quantum interference and entanglement effects
- Non-linear transformations through quantum gates

CHALLENGES:
- Shot noise in measurements → gradient instability
- Barren plateaus → vanishing gradients
- Computational cost: O(exp(n_qubits)) simulation
- Limited expressivity with shallow circuits on NISQ devices
"""


import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from typing import Optional, Tuple

class QuantumEmbedding(nn.Module):
    """
    Quantum Embedding Layer using Parameterized Quantum Circuits
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        n_qubits: int = 6,
        n_layers: int = 3,
        shots: int = 1000,
        device_name: str = "default.qubit"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        
        # Quantum device setup
        self.qdev = qml.device(device_name, wires=n_qubits, shots=shots)
        
        # Parameters for PQC
        # Shape: (vocab_size, n_layers, n_qubits, 2) for RY and RZ rotations
        self.quantum_params = nn.Parameter(
            torch.randn(vocab_size, n_layers, n_qubits, 2) * 0.1
        )
        
        # Classical post-processing layer
        self.post_process = nn.Linear(n_qubits, embedding_dim)
        
        # Create quantum circuit
        self.qcircuit = self._create_quantum_circuit()
    
    def _create_quantum_circuit(self):
        """Create parameterized quantum circuit"""
        
        @qml.qnode(self.qdev, interface="torch", diff_method="parameter-shift")
        def circuit(params):
            # State preparation and parameterized layers
            for layer in range(self.n_layers):
                # Single qubit rotations
                for qubit in range(self.n_qubits):
                    qml.RY(params[layer, qubit, 0], wires=qubit)
                    qml.RZ(params[layer, qubit, 1], wires=qubit)
                
                # Entangling layer (except last layer)
                if layer < self.n_layers - 1:
                    for qubit in range(self.n_qubits - 1):
                        qml.CNOT(wires=[qubit, qubit + 1])
                    # Ring connectivity
                    if self.n_qubits > 2:
                        qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # Measurements - expectation values of Pauli-Z
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return circuit
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum embedding
        
        Args:
            token_ids: Tensor of shape (batch_size, seq_len)
            
        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len = token_ids.shape
        embeddings = []
        
        for batch_idx in range(batch_size):
            batch_embeddings = []
            
            for seq_idx in range(seq_len):
                token_id = token_ids[batch_idx, seq_idx].item()
                
                # Get quantum parameters for this token
                params = self.quantum_params[token_id]
                
                # Execute quantum circuit
                quantum_features = self.qcircuit(params)
                
                # Convert to tensor and apply post-processing
                quantum_tensor = torch.stack(quantum_features)
                classical_embedding = self.post_process(quantum_tensor)
                
                batch_embeddings.append(classical_embedding)
            
            embeddings.append(torch.stack(batch_embeddings))
        
        return torch.stack(embeddings)


class OptimizedQuantumEmbedding(nn.Module):
    """
    Optimized version with batched quantum execution
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        n_qubits: int = 6,
        n_layers: int = 3,
        shots: int = 1000
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        
        # Prefer lightning.qubit for speed; fall back to default.qubit if unavailable
        try:
            self.qdev = qml.device("lightning.qubit", wires=n_qubits)
        except Exception:
            self.qdev = qml.device("default.qubit", wires=n_qubits)
        
        # Quantum parameters
        self.quantum_params = nn.Parameter(
            torch.randn(vocab_size, n_layers * n_qubits * 2) * 0.1
        )
        
        # Post-processing
        self.post_process = nn.Linear(n_qubits, embedding_dim)
        self.activation = nn.Tanh()
        
        # Pre-compute quantum circuits for all tokens (cache)
        self._quantum_cache = {}
        
    def _get_quantum_features(self, token_id: int) -> torch.Tensor:
        """Get quantum features for a specific token with caching"""
        
        if token_id in self._quantum_cache:
            return self._quantum_cache[token_id]
        
        @qml.qnode(self.qdev, interface="torch", diff_method="parameter-shift")
        def circuit(params):
            # Reshape parameters
            params_reshaped = params.reshape(self.n_layers, self.n_qubits, 2)
            
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    qml.RY(params_reshaped[layer, qubit, 0], wires=qubit)
                    qml.RZ(params_reshaped[layer, qubit, 1], wires=qubit)
                
                if layer < self.n_layers - 1:
                    for qubit in range(self.n_qubits - 1):
                        qml.CNOT(wires=[qubit, qubit + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        params = self.quantum_params[token_id]
        quantum_features = circuit(params)
        result = torch.stack(quantum_features)
        
        # Cache result without detaching so autograd can track operations
        self._quantum_cache[token_id] = result
        
        return result
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Clear cache at the start of each forward pass so that cached
        # tensors do not hold references to computation graphs from
        # previous iterations.
        self.clear_cache()

        batch_size, seq_len = token_ids.shape
        
        # Collect all unique token IDs for batch processing
        unique_tokens = torch.unique(token_ids)
        
        # Process quantum features for unique tokens
        quantum_features_dict = {}
        for token_id in unique_tokens:
            quantum_features_dict[token_id.item()] = self._get_quantum_features(
                token_id.item()
            )
        
        # Build embeddings
        embeddings = []
        for batch_idx in range(batch_size):
            batch_embeddings = []
            for seq_idx in range(seq_len):
                token_id = token_ids[batch_idx, seq_idx].item()
                quantum_features = quantum_features_dict[token_id]
                
                # Apply post-processing
                embedding = self.activation(self.post_process(quantum_features))
                batch_embeddings.append(embedding)
            
            embeddings.append(torch.stack(batch_embeddings))
        
        return torch.stack(embeddings)
    
    def clear_cache(self):
        """Clear quantum feature cache"""
        self._quantum_cache.clear()
