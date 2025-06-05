# Hybrid Quantum-Classical Transformer Architecture Specification

## Executive Summary

This document outlines the complete architecture for a hybrid quantum-classical transformer that replaces traditional embedding layers and attention mechanisms with quantum alternatives. The design is optimized for NISQ-era quantum computers while maintaining compatibility with classical deep learning frameworks.

## 1. Overall Architecture

### 1.1 High-Level Flow

Input Tokens → Quantum Embedding → Classical Processing → Quantum Attention → Classical Output

### 1.2 Key Components

- **Quantum Embedding Layer**: Replaces classical word embeddings
- **Hybrid Attention Mechanism**: Quantum-enhanced multi-head attention
- **Classical Transformer Blocks**: Standard feed-forward and normalization layers
- **Quantum-Classical Interface**: Manages state preparation and measurement

## 2. Quantum Embedding Layer

### 2.1 Architecture

**Purpose**: Transform discrete token IDs into quantum-enhanced continuous representations

**Input**: Token IDs \( t \in \{1, 2, ..., V\} \) where \( V \) is vocabulary size
**Output**: Classical vectors \( \mathbf{e} \in \mathbb{R}^{d_{model}} \)

### 2.2 Quantum Circuit Design

#### 2.2.1 State Preparation

- **Qubits Required**: \( n_q = \lceil \log_2(d_{model}) \rceil \)
- **Encoding Method**: Angle encoding for continuous features
- **Circuit Depth**: 3-5 layers (NISQ-compatible)

#### 2.2.2 Parameterized Quantum Circuit (PQC)

|0⟩^n → RY(θ₁) → RZ(φ₁) → CNOT → RY(θ₂) → RZ(φ₂) → ... → Measurement

**Parameters**:
- \( \theta_i, \phi_i \) are trainable parameters
- Total parameters: \( 2 \times n_q \times depth \times V \)
- Parameter initialization: Xavier/He initialization adapted for quantum

#### 2.2.3 Measurement Strategy

- **Observable**: Pauli-Z measurements on each qubit
- **Expectation Values**: \( \langle Z_i \rangle \) for qubit \( i \)
- **Output Transformation**: \( e_i = \tanh(\langle Z_i \rangle) \)

### 2.3 Training Protocol

- **Gradient Estimation**: Parameter-shift rule
- **Shot Budget**: 1000-5000 shots per parameter
- **Optimization**: Adam optimizer with quantum-aware learning rates

## 3. Quantum Attention Mechanism

### 3.1 Architecture Overview

**Purpose**: Compute attention weights using quantum interference and superposition

**Input**: Query (Q), Key (K), Value (V) matrices from classical linear projections
**Output**: Classical attention output \( \mathbf{O} \in \mathbb{R}^{seq\_len \times d_{model}} \)

### 3.2 Quantum Attention Circuit

#### 3.2.1 Multi-Head Design

- **Heads**: \( h = 8 \) (standard transformer)
- **Qubits per Head**: \( n_{head} = \lceil \log_2(seq\_len) \rceil + 2 \)
- **Total Qubits**: \( h \times n_{head} \)

#### 3.2.2 State Preparation for Q, K, V

# Pseudocode for state preparation
def prepare_qkv_state(Q, K, V, head_idx):
    # Normalize to unit vectors
    q_norm = Q / ||Q||₂
    k_norm = K / ||K||₂
    v_norm = V / ||V||₂

    # Encode as quantum amplitudes
    |ψ_q⟩ = Σᵢ q_norm[i] |i⟩
    |ψ_k⟩ = Σᵢ k_norm[i] |i⟩
    |ψ_v⟩ = Σᵢ v_norm[i] |i⟩

#### 3.2.3 Quantum Inner Product Computation

**SWAP Test Circuit** for computing \( \langle\psi_q|\psi_k\rangle \):

Ancilla: |0⟩ ——— H ——— • ——— H ——— M
                      |
Query:   |ψ_q⟩ ——————— SWAP ————————
                      |
Key:     |ψ_k⟩ ——————— SWAP ————————

**Attention Weight Calculation**:
\[
\alpha_{ij} = \frac{\exp(\beta \cdot \text{Re}(\langle\psi_{q_i}|\psi_{k_j}\rangle))}{\sum_k \exp(\beta \cdot \text{Re}(\langle\psi_{q_i}|\psi_{k_k}\rangle))}
\]

Where \( \beta \) is a trainable temperature parameter.

### 3.3 Value Aggregation

- **Quantum Amplitude Encoding**:
  - Encode attention weights as quantum amplitudes
  - Apply controlled rotations based on Value vectors
  - Measure expectation values for final output

## 4. Classical Components

### 4.1 Standard Transformer Blocks

- **Layer Normalization**: Pre-LN configuration
- **Feed-Forward Networks**: 2-layer MLP with GELU activation
- **Residual Connections**: Standard skip connections
- **Position Encoding**: Sinusoidal encoding (classical)

### 4.2 Output Layer

- **Linear Projection**: \( d_{model} \rightarrow vocab\_size \)
- **Softmax**: Standard probability distribution
- **Loss Function**: Cross-entropy loss

## 5. Implementation Requirements

### 5.1 Quantum Hardware Specifications

**NISQ Requirements**:
- **Minimum Qubits**: 20-30 qubits
- **Gate Fidelity**: > 99% for single-qubit, > 95% for two-qubit
- **Coherence Time**: > 100μs
- **Connectivity**: All-to-all or near all-to-all

**Simulator Requirements**:
- **Classical Simulation**: Up to 25 qubits
- **Hardware**: High-memory systems (32+ GB RAM)
- **Framework**: Qiskit, Cirq, or PennyLane

### 5.2 Software Stack

Application Layer:    Custom Hybrid Transformer
Quantum ML Layer:     PennyLane/TensorFlow Quantum
Classical ML Layer:   PyTorch/TensorFlow
Quantum Framework:    Qiskit/Cirq
Hardware Layer:       IBM Quantum/Google Quantum AI

### 5.3 Training Configuration

**Hyperparameters**:
- **Learning Rate**: 1e-4 (classical), 5e-3 (quantum)
- **Batch Size**: 16-32 (limited by quantum circuit depth)
- **Sequence Length**: 128-512 tokens
- **Model Dimension**: 256-512
- **Quantum Shots**: 1000-5000 per forward pass

**Training Strategy**:
1. **Pre-training Phase**: Classical components only
2. **Quantum Integration**: Gradual replacement of classical layers
3. **Fine-tuning**: End-to-end quantum-classical optimization

## 6. Evaluation Metrics

### 6.1 Performance Metrics

- **Perplexity**: Language modeling capability
- **BLEU/ROUGE**: Translation/summarization tasks
- **Classification Accuracy**: Downstream tasks

### 6.2 Quantum-Specific Metrics

- **Quantum Volume**: Circuit complexity measure
- **Fidelity**: Quantum state preparation accuracy
- **Gate Count**: NISQ-compatibility assessment
- **Shot Efficiency**: Quantum resource utilization

### 6.3 Efficiency Metrics

- **Training Time**: Wall-clock time comparison
- **Inference Speed**: Tokens per second
- **Energy Consumption**: Power usage analysis
- **Scalability**: Performance vs. model size

## 7. Experimental Protocol

### 7.1 Baseline Comparisons

- **Classical Transformer**: Standard BERT/GPT architecture
- **Hybrid Variants**: Different quantum component combinations
- **Ablation Studies**: Individual component analysis

### 7.2 Dataset Requirements

- **Small Scale**: Penn Treebank, WikiText-2 (debugging)
- **Medium Scale**: BookCorpus, OpenWebText (validation)
- **Large Scale**: Common Crawl (production testing)

### 7.3 Implementation Phases

Current progress based on the repository:

- [x] **Phase 1**: Quantum embedding layer implemented (`src/quantum_embedding.py` and `tests/test_quantum_embedding.py`)
- [x] **Phase 2**: Quantum attention integration implemented (`src/quantum_attention.py` and `tests/test_quantum_attention_layer.py`)
- [x] **Phase 3**: Full hybrid architecture implemented (`src/hybrid_transformer.py`)
- [ ] **Phase 4**: Optimization and scaling
- [ ] **Phase 5**: Evaluation on a real dataset

## 8. Expected Challenges and Mitigation

### 8.1 Technical Challenges

- **Quantum Noise**: Error mitigation techniques
- **Limited Coherence**: Circuit optimization
- **Classical-Quantum Interface**: Efficient data transfer
- **Gradient Estimation**: Parameter-shift rule implementation

### 8.2 Mitigation Strategies

- **Noise-Aware Training**: Robust optimization techniques
- **Circuit Compression**: Gate synthesis and optimization
- **Hybrid Batching**: Efficient quantum-classical data flow
- **Variance Reduction**: Advanced gradient estimation
