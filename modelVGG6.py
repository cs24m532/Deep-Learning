import torch
import torch.nn as nn

class VGG6(nn.Module):
    """
    A lightweight VGG-style CNN for CIFAR-10 classification (6 convolutional layers).

    Architecture:
        Input: 3 x 32 x 32 (CIFAR-10 images)
        [Conv2d → Activation] x 2 → MaxPool
        [Conv2d → Activation] x 2 → MaxPool
        [Conv2d → Activation] x 1 → MaxPool
        Flatten → Linear → Activation → Dropout → Linear → Output
    """

    def __init__(self, num_classes=10, activation='relu'):
        super().__init__()

        # Get the chosen activation function dynamically
        act = self._get_activation(activation)

        # -------------------------------
        # FEATURE EXTRACTOR (Convolutional Layers)
        # -------------------------------
        self.features = nn.Sequential(
            # --- Block 1 ---
            # Input: (3 x 32 x 32)
            nn.Conv2d(3, 64, kernel_size=3, padding=1),   # Output: (64 x 32 x 32)
            act,                                          # Non-linearity
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Output: (64 x 32 x 32)
            act,
            nn.MaxPool2d(2, 2),                           # Downsample: (64 x 16 x 16)

            # --- Block 2 ---
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Output: (128 x 16 x 16)
            act,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# Output: (128 x 16 x 16)
            act,
            nn.MaxPool2d(2, 2),                           # Downsample: (128 x 8 x 8)

            # --- Block 3 ---
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# Output: (256 x 8 x 8)
            act,
            nn.MaxPool2d(2, 2),                           # Downsample: (256 x 4 x 4)
        )

        # -------------------------------
        # CLASSIFIER (Fully Connected Layers)
        # -------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # Flatten 256x4x4 → 4096
            nn.Linear(256 * 4 * 4, 512),                  # Fully connected: 4096 → 512
            act,                                          # Apply chosen activation
            nn.Dropout(0.5),                              # Regularization (drop 50% neurons)
            nn.Linear(512, num_classes),                  # Output layer: 512 → num_classes
        )

    def _get_activation(self, name):
        """
        Returns the activation function layer based on user input.

        Supported:
            'relu', 'sigmoid', 'tanh', 'silu'/'swish', 'gelu'
        Default: ReLU
        """
        name = name.lower()
        if name == 'relu': return nn.ReLU(inplace=True)
        if name == 'sigmoid': return nn.Sigmoid()
        if name == 'tanh': return nn.Tanh()
        if name in ('silu', 'swish'): return nn.SiLU()
        if name == 'gelu': return nn.GELU()
        return nn.ReLU(inplace=True)  # Default activation

    def forward(self, x):
        """
        Defines the forward propagation through the network.
        """
        x = self.features(x)      # Extract spatial features using conv blocks
        x = self.classifier(x)    # Flatten and classify with fully connected layers
        return x
