import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import sys

from src.classical_transformer import ClassicalTransformer
from src.hybrid_transformer import HybridTransformer


class TextDataset(Dataset):
    """Tokenized text dataset with padding."""

    def __init__(self, texts, labels, vocab, max_len=32):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.word2idx["<UNK>"] = len(vocab)
        self.word2idx["<PAD>"] = len(vocab) + 1

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].lower().split()[: self.max_len]
        label = self.labels[idx]
        tokens = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in text]
        while len(tokens) < self.max_len:
            tokens.append(self.word2idx["<PAD>"])
        return torch.tensor(tokens), torch.tensor(label, dtype=torch.long)


def create_datasets(max_len=32):
    categories = [
        "alt.atheism",
        "comp.graphics",
        "sci.med",
        "soc.religion.christian",
    ]
    train = fetch_20newsgroups(
        subset="train", categories=categories, remove=("headers", "footers", "quotes")
    )
    test = fetch_20newsgroups(
        subset="test", categories=categories, remove=("headers", "footers", "quotes")
    )
    vectorizer = CountVectorizer(max_features=150, stop_words="english")
    vectorizer.fit(train.data)
    vocab = list(vectorizer.vocabulary_.keys())

    train_dataset = TextDataset(train.data[:1000], train.target[:1000], vocab, max_len)
    test_dataset = TextDataset(test.data[:400], test.target[:400], vocab, max_len)
    return train_dataset, test_dataset, len(vocab) + 2


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            preds = logits.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / max(1, total)


def train():
    train_ds, test_ds, vocab_size = create_datasets()
    batch_size = 8
    embed_dim = 32
    num_classes = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    classical_model = ClassicalTransformer(vocab_size, embed_dim, num_classes).to(device)
    hybrid_model = HybridTransformer(vocab_size, embed_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    classical_opt = optim.Adam(classical_model.parameters(), lr=1e-3)
    hybrid_opt = optim.Adam(hybrid_model.parameters(), lr=5e-3)

    epochs = 3
    classical_losses = []
    hybrid_losses = []
    for epoch in range(epochs):
        classical_loss_sum = 0.0
        hybrid_loss_sum = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            classical_opt.zero_grad()
            c_out = classical_model(data)
            c_loss = criterion(c_out, target)
            c_loss.backward()
            classical_opt.step()
            classical_loss_sum += c_loss.item()

            if batch_idx % 2 == 0:
                hybrid_opt.zero_grad()
                h_out = hybrid_model(data)
                h_loss = criterion(h_out, target)
                h_loss.backward()
                hybrid_opt.step()
                hybrid_loss_sum += h_loss.item()

        classical_losses.append(classical_loss_sum / len(train_loader))
        hybrid_losses.append(hybrid_loss_sum / (len(train_loader) // 2))
        print(
            f"Epoch {epoch+1}/{epochs} - Classical Loss: {classical_losses[-1]:.4f} - Hybrid Loss: {hybrid_losses[-1]:.4f}"
        )

    hybrid_acc = evaluate(hybrid_model, test_loader, device)
    classical_acc = evaluate(classical_model, test_loader, device)
    print(f"Test Accuracy - Classical: {classical_acc:.3f} - Hybrid: {hybrid_acc:.3f}")

    plt.figure(figsize=(10, 6))
    plt.plot(classical_losses, label="Classical")
    plt.plot(hybrid_losses, label="Hybrid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss - Classical vs Hybrid Transformer")
    plt.legend()
    plt.grid(True)
    plt.savefig("hybrid_training_comparison.png")
    plt.close()

    return classical_model, hybrid_model


if __name__ == "__main__":
    train()
