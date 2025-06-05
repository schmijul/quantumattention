# QuantumAttention

This repository explores a **hybrid quantum\-classical transformer** where classical embedding and attention layers can be replaced with quantum counterparts. The overall architecture is inspired by the plan in `plan.md` and follows the flow:

```
Input Tokens → Quantum Embedding → Classical Processing → Quantum Attention → Classical Output
```

Key components include a quantum embedding layer implemented with **parameterized quantum circuits (PQCs)**, a hybrid attention mechanism and standard transformer blocks. The goal is to investigate how quantum features might enrich language models on near term (NISQ) hardware.

## Quantum Embedding Layer

Relevant details from the plan:

- Qubits required: `n_q = ceil(log2(d_model))`
- Angle encoding and 3\-5 layers of RY/RZ rotations with CNOT entanglement
- Pauli\-Z measurements provide features which are transformed via a small linear layer
- Gradients are computed using the parameter\-shift rule with a shot budget of roughly 1000\-5000

The implementation can be found in [`src/quantum_embedding.py`](src/quantum_embedding.py). The file defines a basic `QuantumEmbedding` and an `OptimizedQuantumEmbedding` that caches quantum evaluations for efficiency.

## Training Comparison

*   `train_compare.py`:
    *   Conducts an experiment comparing a `ClassicalTransformer` against a `HybridQuantumEmbeddingTransformer` on a synthetic sentiment text classification task.
    *   **Key Finding:** The `HybridQuantumEmbeddingTransformer` ultimately achieves **superior performance metrics (notably lower final training and validation loss, and comparable final validation accuracy)** compared to the classical model, when using a similar number of total parameters and identical training hyperparameters.
    *   **Learning Dynamics:**
        *   The classical embedding model converges more rapidly in the initial epochs.
        *   The quantum-enhanced model, while exhibiting slower initial convergence, demonstrates robust learning behavior and surpasses the classical model's loss metrics by the end of training, achieving higher training accuracy.
    *   **Computational Cost:** Training the hybrid quantum model is significantly slower per epoch due to the overhead of quantum circuit simulation.
    *   **Output:** The script generates `quantum_embedding_comparison.png`, visually presenting the loss and accuracy curves for both models, illustrating these dynamics.

![Training comparison](quantum_embedding_comparison.png)

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Then run the example training:

```bash
python train_quantum_embedding.py
```

This will train both models for a few epochs and save the comparison plot.
