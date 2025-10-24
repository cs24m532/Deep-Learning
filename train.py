# ==============================
# VGG6 CIFAR-10 Training Script
# ==============================
# Implements a simple training loop for a custom VGG6 model on CIFAR-10.
# Includes support for different optimizers, activations, reproducibility, 
# and optional experiment logging using Weights & Biases (wandb).

import argparse, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from modelVGG6 import VGG6                     # Custom VGG6 model
from utils import get_cifar10_loaders       # Utility to load CIFAR-10 dataset
from tqdm import tqdm                       # For progress bars

# Try to import wandb (Weights & Biases) for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


# -----------------------------
# Set all random seeds for reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Train for one epoch
# -----------------------------
def train_one_epoch(modelVGG6, loader, criterion, optimizer, device):
    modelVGG6.train()                      # Set model to training mode
    running_loss, correct, total = 0, 0, 0

    # Loop over all training batches
    for x, y in tqdm(loader, desc='train', leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()           # Reset gradients
        out = modelVGG6(x)                  # Forward pass
        loss = criterion(out, y)        # Compute loss
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update parameters

        # Track metrics
        running_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return running_loss / total, correct / total


# -----------------------------
# Evaluate model on validation/test data
# -----------------------------
def evaluate(modelVGG6, loader, criterion, device):
    modelVGG6.eval()                       # Set model to evaluation mode
    running_loss, correct, total = 0, 0, 0

    with torch.no_grad():              # Disable gradients during evaluation
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = modelVGG6(x)
            loss = criterion(out, y)

            # Track metrics
            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return running_loss / total, correct / total


# -----------------------------
# Get optimizer by name
# -----------------------------
def get_optimizer(name, params, lr, momentum=0.9):
    """
    Returns a PyTorch optimizer based on user-specified name.
    """
    name = name.lower()
    if name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=momentum)
    if name == 'nesterov':
        return optim.SGD(params, lr=lr, momentum=momentum, nesterov=True)
    if name == 'adam':
        return optim.Adam(params, lr=lr)
    if name == 'nadam':
        return optim.NAdam(params, lr=lr)
    if name == 'adagrad':
        return optim.Adagrad(params, lr=lr)
    if name == 'rmsprop':
        return optim.RMSprop(params, lr=lr, momentum=momentum)
    raise ValueError(f'Unknown optimizer {name}')


# -----------------------------
# Main training function
# -----------------------------
def main():
    # Argument parser for CLI options
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--activation', type=str, default='relu')
    p.add_argument('--optimizer', type=str, default='sgd')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--project', type=str, default='vgg6_cifar10')
    p.add_argument('--save_dir', type=str, default='./checkpoints')
    p.add_argument('--no_wandb', action='store_true')
    a = p.parse_args()

    # Set seed and device
    set_seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.save_dir, exist_ok=True)

    # Load CIFAR-10 data loaders
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=a.batch_size)

    # Create model, loss, optimizer
    modelVGG6 = VGG6(num_classes=10, activation=a.activation).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(a.optimizer, modelVGG6.parameters(), lr=a.lr)

    # Initialize wandb (if available and not disabled)
    if WANDB_AVAILABLE and not a.no_wandb:
        wandb.init(project=a.project, config=vars(a))
        wandb.watch(modelVGG6)

    # -----------------------------
    # Training loop
    # -----------------------------
    best_val = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, a.epochs + 1):
        # Train and evaluate
        tl, ta = train_one_epoch(modelVGG6, train_loader, criterion, optimizer, device)
        vl, va = evaluate(modelVGG6, val_loader, criterion, device)

        # Print progress
        print(f'Epoch {epoch}/{a.epochs}  Train Acc: {ta:.4f}  Val Acc: {va:.4f}')

        # Save metrics
        history['train_loss'].append(tl)
        history['train_acc'].append(ta)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)

        # Log metrics to wandb
        if WANDB_AVAILABLE and not a.no_wandb:
            wandb.log({'epoch': epoch, 'train_loss': tl, 'train_acc': ta,
                       'val_loss': vl, 'val_acc': va})

        # Save best model checkpoint
        if va > best_val:
            best_val = va
            save = os.path.join(
                a.save_dir, f'best_{a.activation}_{a.optimizer}_bs{a.batch_size}_lr{a.lr}.pth'
            )
            torch.save({'modelVGG6_state_dict': modelVGG6.state_dict(), 'val_acc': va}, save)

    # Log best validation accuracy to wandb summary
    if WANDB_AVAILABLE and not a.no_wandb:
        wandb.run.summary['best_val_acc'] = best_val


# -----------------------------
# Entry point
# -----------------------------
if __name__ == '__main__':
    main()
