"""
Training Comparison: Classical vs Hybrid Quantum Embedding Transformer
Uses synthetic sentiment data with robust error handling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from pathlib import Path
import random


from src.classical_transformer import ClassicalTransformer
from src.hybrid_quantum_embedding_transformer import HybridQuantumEmbeddingTransformer

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- Hyperparameters ---
BATCH_SIZE = 16
EMBED_DIM = 8
N_CLASSES = 2  # Binary sentiment
N_EPOCHS = 10
SEQ_LEN = 6
VOCAB_SIZE = 20
N_QUBITS = 4  # Smaller for faster training
N_LAYERS = 2
SHOTS = 100   # Fewer shots for faster training
TRAIN_SIZE = 800
VAL_SIZE = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Synthetic Data Generation ---
class SyntheticSentimentDataset(Dataset):
    """
    Generate synthetic sentiment data:
    - Positive: sequences with more high-value tokens (10-19)
    - Negative: sequences with more low-value tokens (1-9)
    """
    
    def __init__(self, size, seq_len, vocab_size):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = self._generate_data()
    
    def _generate_data(self):
        data = []
        for _ in range(self.size):
            # Random label
            label = random.randint(0, 1)
            
            if label == 1:  # Positive
                # More high-value tokens
                sequence = [random.randint(10, self.vocab_size-1) for _ in range(self.seq_len//2)]
                sequence += [random.randint(1, self.vocab_size-1) for _ in range(self.seq_len - len(sequence))]
            else:  # Negative
                # More low-value tokens
                sequence = [random.randint(1, 9) for _ in range(self.seq_len//2)]
                sequence += [random.randint(1, self.vocab_size-1) for _ in range(self.seq_len - len(sequence))]
            
            # Shuffle to make it less obvious
            random.shuffle(sequence)
            
            # Pad with zeros if needed
            if len(sequence) < self.seq_len:
                sequence += [0] * (self.seq_len - len(sequence))
            
            data.append((torch.tensor(sequence, dtype=torch.long), label))
        
        return data
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]

# Create datasets
print("Generating synthetic sentiment data...")
train_dataset = SyntheticSentimentDataset(TRAIN_SIZE, SEQ_LEN, VOCAB_SIZE)
val_dataset = SyntheticSentimentDataset(VAL_SIZE, SEQ_LEN, VOCAB_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Vocab size: {VOCAB_SIZE}")
print(f"Sequence length: {SEQ_LEN}")

# --- Training Utilities ---
def train_epoch(model, loader, criterion, optimizer, model_name="Model"):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    failed_batches = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        try:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  {model_name}: NaN/Inf loss in batch {batch_idx}, skipping...")
                failed_batches += 1
                continue
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
        except Exception as e:
            print(f"⚠️  {model_name}: Error in batch {batch_idx}: {e}")
            failed_batches += 1
            continue
    
    if failed_batches > 0:
        print(f"⚠️  {model_name}: {failed_batches}/{len(loader)} batches failed")
    
    if total == 0:
        return float('inf'), 0.0
    
    return total_loss / (len(loader) - failed_batches), correct / total

def eval_epoch(model, loader, criterion, model_name="Model"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    failed_batches = 0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            try:
                out = model(x)
                loss = criterion(out, y)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    failed_batches += 1
                    continue
                
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
            except Exception as e:
                print(f"⚠️  {model_name}: Error in validation batch {batch_idx}: {e}")
                failed_batches += 1
                continue
    
    if total == 0:
        return float('inf'), 0.0
    
    return total_loss / (len(loader) - failed_batches), correct / total

# --- Main Training Loop ---
def run_experiment(model_class, model_name, **model_kwargs):
    print(f"\n{'='*50}")
    print(f"🚀 Training {model_name}")
    print(f"{'='*50}")
    
    try:
        model = model_class(VOCAB_SIZE, EMBED_DIM, N_CLASSES, **model_kwargs).to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        
        # Show quantum info if available
        if hasattr(model, 'get_quantum_info'):
            quantum_info = model.get_quantum_info()
            print(f"🔬 Quantum parameters: {quantum_info['quantum_parameters']:,}")
            print(f"🔬 Classical parameters: {quantum_info['classical_parameters']:,}")
            print(f"🔬 Quantum ratio: {quantum_info['quantum_ratio']:.1%}")
            print(f"🔬 Qubits: {quantum_info['n_qubits']}, Layers: {quantum_info['n_layers']}")
        
        # Test forward pass first
        print("🧪 Testing forward pass...")
        test_x = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN)).to(device)
        with torch.no_grad():
            test_out = model(test_x)
            print(f"   ✅ Forward pass works: {test_out.shape}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(N_EPOCHS):
            t0 = time.time()
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, model_name)
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, model_name)
            
            # Check for training failure
            if train_loss == float('inf') or val_loss == float('inf'):
                print(f"❌ Training failed at epoch {epoch+1}")
                return None
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            epoch_time = time.time() - t0
            
            print(f"Epoch {epoch+1:2d}/{N_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'model': model
        }
        
    except Exception as e:
        print(f"❌ Error training {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_results(classical_results, hybrid_results=None):
    """Plot training results comparison."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs = range(1, len(classical_results['train_losses']) + 1)
    
    # Training Loss
    ax1.plot(epochs, classical_results['train_losses'], 'b-', label='Classical', marker='o', linewidth=2)
    if hybrid_results:
        hybrid_epochs = range(1, len(hybrid_results['train_losses']) + 1)
        ax1.plot(hybrid_epochs, hybrid_results['train_losses'], 'g-', label='Quantum Embedding', marker='s', linewidth=2)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation Loss
    ax2.plot(epochs, classical_results['val_losses'], 'b-', label='Classical', marker='o', linewidth=2)
    if hybrid_results:
        ax2.plot(hybrid_epochs, hybrid_results['val_losses'], 'g-', label='Quantum Embedding', marker='s', linewidth=2)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax3.plot(epochs, classical_results['train_accs'], 'b-', label='Classical', marker='o', linewidth=2)
    if hybrid_results:
        ax3.plot(hybrid_epochs, hybrid_results['train_accs'], 'g-', label='Quantum Embedding', marker='s', linewidth=2)
    ax3.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Validation Accuracy
    ax4.plot(epochs, classical_results['val_accs'], 'b-', label='Classical', marker='o', linewidth=2)
    if hybrid_results:
        ax4.plot(hybrid_epochs, hybrid_results['val_accs'], 'g-', label='Quantum Embedding', marker='s', linewidth=2)
    ax4.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quantum_embedding_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved as 'quantum_embedding_comparison.png'")
    plt.show()

def print_summary(classical_results, hybrid_results=None):
    """Print training summary."""
    print(f"\n{'='*60}")
    print("📈 TRAINING SUMMARY")
    print(f"{'='*60}")
    
    print(f"🔵 Classical Transformer:")
    print(f"   Final Train Loss: {classical_results['train_losses'][-1]:.4f}")
    print(f"   Final Val Loss:   {classical_results['val_losses'][-1]:.4f}")
    print(f"   Final Train Acc:  {classical_results['train_accs'][-1]:.4f}")
    print(f"   Final Val Acc:    {classical_results['val_accs'][-1]:.4f}")
    
    if hybrid_results:
        print(f"\n🟢 Quantum Embedding Transformer:")
        print(f"   Final Train Loss: {hybrid_results['train_losses'][-1]:.4f}")
        print(f"   Final Val Loss:   {hybrid_results['val_losses'][-1]:.4f}")
        print(f"   Final Train Acc:  {hybrid_results['train_accs'][-1]:.4f}")
        print(f"   Final Val Acc:    {hybrid_results['val_accs'][-1]:.4f}")
        
        # Performance comparison
        print(f"\n📊 Performance Comparison:")
        val_acc_diff = hybrid_results['val_accs'][-1] - classical_results['val_accs'][-1]
        val_loss_diff = hybrid_results['val_losses'][-1] - classical_results['val_losses'][-1]
        
        print(f"   Val Accuracy Difference: {val_acc_diff:+.4f}")
        print(f"   Val Loss Difference:     {val_loss_diff:+.4f}")
        
        if val_acc_diff > 0.01:
            print("   🎉 Quantum embedding performs better!")
        elif val_acc_diff < -0.01:
            print("   🔵 Classical model performs better!")
        else:
            print("   🤝 Similar performance!")
    else:
        print("\n🟢 Quantum Embedding Transformer: Training failed")

if __name__ == "__main__":
    print("🧪 Classical vs Quantum Embedding Transformer Training Comparison")
    print(f"📊 Dataset: Synthetic sentiment ({TRAIN_SIZE} train, {VAL_SIZE} val)")
    print(f"🔧 Hyperparameters: {N_EPOCHS} epochs, batch_size={BATCH_SIZE}")
    
    # Train Classical Model
    classical_results = run_experiment(ClassicalTransformer, "Classical Transformer")
    
    # Train Quantum Embedding Model
    hybrid_results = run_experiment(
        HybridQuantumEmbeddingTransformer, 
        "Quantum Embedding Transformer",
        n_qubits=N_QUBITS, 
        n_layers=N_LAYERS, 
        shots=SHOTS
    )
    
    # Results
    if classical_results:
        print_summary(classical_results, hybrid_results)
        plot_results(classical_results, hybrid_results)
    else:
        print("❌ All training failed!")