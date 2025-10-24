import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

def get_cifar10_loaders(batch_size=128, num_workers=4, augment=True, val_split=5000, seed=42):
    """
    Returns train, validation, and test DataLoaders for the CIFAR-10 dataset.

    Args:
        batch_size (int): Number of samples per batch to load.
        num_workers (int): Number of subprocesses to use for data loading.
        augment (bool): Whether to apply data augmentation to the training set.
        val_split (int): Number of samples to use for the validation set.
        seed (int): Random seed for reproducible train/val splitting.

    Returns:
        (train_loader, val_loader, test_loader): DataLoaders for training, validation, and test sets.
    """

    # Mean and standard deviation of CIFAR-10 (used for normalization)
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    # Build the training transform pipeline
    train_transforms = []

    # Apply data augmentation if enabled
    if augment:
        train_transforms += [
            transforms.RandomCrop(32, padding=4),   # Randomly crop image with padding
            transforms.RandomHorizontalFlip()       # Randomly flip image horizontally
        ]

    # Convert images to tensors and normalize
    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    # Compose all training transformations
    train_tf = transforms.Compose(train_transforms)

    # Define test/validation transforms (no augmentation)
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load CIFAR-10 training and test datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_tf
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_tf
    )

    # Set random seed for reproducibility of the train/validation split
    torch.manual_seed(seed)

    # Split the training set into training and validation subsets
    val_size = val_split
    train_size = len(trainset) - val_size
    train_data, val_data = random_split(trainset, [train_size, val_size])

    # Create DataLoaders for each dataset
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Return all three DataLoaders
    return train_loader, val_loader, test_loader
