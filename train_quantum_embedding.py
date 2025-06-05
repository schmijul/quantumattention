# train_quantum_embedding.py

"""
Quantum vs Classical Embedding Training Comparison Script

OVERVIEW:
- Compares quantum parameterized circuits (PQC) against standard PyTorch embeddings
- Uses text classification as benchmark task
- Measures training loss convergence and performance differences

DATA:
- Dataset: 20newsgroups (4 categories: atheism, graphics, medicine, religion)
- Size: 1000 samples, 150-word vocabulary
- Preprocessing: Tokenization, padding to 32 tokens, UNK/PAD handling

MODELS:
- Classical: Standard nn.Embedding + MLP classifier
- Quantum: OptimizedQuantumEmbedding (6 qubits, 2 layers) + same MLP
- Both use global average pooling and identical classification heads

TRAINING:
- Classical model: Every batch, lr=1e-3
- Quantum model: Every 2nd batch (computational cost), lr=5e-3
- Gradient clipping and linear warm-up for quantum parameters
- 5 epochs, batch_size=8, CrossEntropyLoss
- Outputs training curves and comparison plot

QUANTUM SPECS:
- 6 qubits, 2-layer PQC with RY/RZ rotations + CNOT entangling
- 1000 shots per measurement
- Parameter-shift rule for gradients
- Lightning.qubit simulator backend
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt

from src.quantum_embedding import QuantumEmbedding, OptimizedQuantumEmbedding
from torch.optim.lr_scheduler import LambdaLR


class SimpleTextDataset(Dataset):
    """Simple text classification dataset"""
    
    def __init__(self, texts, labels, vocab, max_len=32):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
        # Create word to index mapping
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.word2idx['<UNK>'] = len(vocab)
        self.word2idx['<PAD>'] = len(vocab) + 1
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx].lower().split()[:self.max_len]
        label = self.labels[idx]
        
        # Convert to indices
        indices = [
            self.word2idx.get(word, self.word2idx['<UNK>']) 
            for word in text
        ]
        
        # Pad sequence
        while len(indices) < self.max_len:
            indices.append(self.word2idx['<PAD>'])
        
        return torch.tensor(indices), torch.tensor(label, dtype=torch.long)


class SimpleClassifier(nn.Module):
    """Simple classifier using quantum embeddings"""
    
    def __init__(self, embedding_layer, embedding_dim, num_classes, max_len):
        super().__init__()
        self.embedding = embedding_layer
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embeddings = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Global average pooling
        pooled = embeddings.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Classification
        logits = self.classifier(pooled)
        return logits


def create_dataset():
    """Create a simple dataset for testing"""
    # Use 20newsgroups dataset (subset)
    categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
    newsgroups = fetch_20newsgroups(
        subset='train', 
        categories=categories, 
        remove=('headers', 'footers', 'quotes')
    )
    
    # Simple preprocessing
    texts = [text[:500] for text in newsgroups.data[:1000]]  # Limit size
    labels = newsgroups.target[:1000]
    
    # Create vocabulary
    vectorizer = CountVectorizer(max_features=150, stop_words='english')
    vectorizer.fit(texts)
    vocab = list(vectorizer.vocabulary_.keys())
    
    return texts, labels, vocab


def train_model():
    """Main training function"""
    
    # Create dataset
    print("Creating dataset...")
    texts, labels, vocab = create_dataset()
    
    # Dataset parameters
    vocab_size = len(vocab) + 2  # +2 for UNK and PAD
    embedding_dim = 32
    max_len = 32
    num_classes = 4
    
    # Create dataset and dataloader
    dataset = SimpleTextDataset(texts, labels, vocab, max_len)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create models - Classical vs Quantum
    print("Creating models...")
    
    # Classical embedding baseline
    classical_embedding = nn.Embedding(vocab_size, embedding_dim)
    classical_model = SimpleClassifier(
        classical_embedding, embedding_dim, num_classes, max_len
    )
    
    # Quantum embedding
    quantum_embedding = OptimizedQuantumEmbedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_qubits=6,
        n_layers=2,
        shots=1000
    )
    quantum_model = SimpleClassifier(
        quantum_embedding, embedding_dim, num_classes, max_len
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    
    classical_optimizer = optim.Adam(classical_model.parameters(), lr=1e-3)
    quantum_optimizer = optim.Adam(quantum_model.parameters(), lr=5e-3)

    # Linear warm-up scheduler for quantum optimizer
    warmup_steps = 10
    lr_lambda = lambda step: min(1.0, (step + 1) / warmup_steps)
    quantum_scheduler = LambdaLR(quantum_optimizer, lr_lambda)
    
    # Training loop
    num_epochs = 5
    
    classical_losses = []
    quantum_losses = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        classical_epoch_loss = 0
        quantum_epoch_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Classical model training
            classical_optimizer.zero_grad()
            classical_output = classical_model(data)
            classical_loss = criterion(classical_output, target)
            classical_loss.backward()
            classical_optimizer.step()
            classical_epoch_loss += classical_loss.item()
            
            # Quantum model training (less frequent due to computational cost)
            if batch_idx % 2 == 0:  # Train every other batch
                quantum_optimizer.zero_grad()
                quantum_output = quantum_model(data)
                quantum_loss = criterion(quantum_output, target)
                quantum_loss.backward()

                # Gradient clipping for quantum parameters
                torch.nn.utils.clip_grad_norm_(
                    quantum_embedding.quantum_params, max_norm=1.0
                )

                quantum_optimizer.step()
                quantum_scheduler.step()
                quantum_epoch_loss += quantum_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Classical Loss: {classical_loss.item():.4f}")
                if batch_idx % 2 == 0:
                    print(f"  Quantum Loss: {quantum_loss.item():.4f}")
        
        classical_losses.append(classical_epoch_loss / len(dataloader))
        quantum_losses.append(quantum_epoch_loss / (len(dataloader) // 2))
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Avg Classical Loss: {classical_losses[-1]:.4f}")
        print(f"  Avg Quantum Loss: {quantum_losses[-1]:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(classical_losses, label='Classical Embedding', marker='o')
    plt.plot(quantum_losses, label='Quantum Embedding', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Classical vs Quantum Embedding Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_comparison.png')
    plt.show()
    
    return classical_model, quantum_model


if __name__ == "__main__":
    classical_model, quantum_model = train_model()
