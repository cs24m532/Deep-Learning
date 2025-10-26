import torch
import torch.nn as nn

class VGG6(nn.Module):
    """
    A lightweight VGG-style CNN for CIFAR-10 classification (6 convolutional layers total).

    Input shape:  (3, 32, 32)
    Output shape: (10,) → logits for CIFAR-10 classes

    Structure:
        [Conv2d → Activation] x 2 → MaxPool
        [Conv2d → Activation] x 2 → MaxPool
        [Conv2d → Activation] x 1 → MaxPool
        Flatten → Linear → Activation → Dropout → Linear
    """

    def __init__(self, num_classes=10, activation='relu'):
        super().__init__()

        # Taking the chosen activation function
        act = self._get_activation(activation)

        # ---------------------------------------------------------------------
        # FEATURE EXTRACTOR
        # ---------------------------------------------------------------------
        # It learns *spatial features* (edges, shapes, textures, etc.)
        # from the input images using convolution and pooling.
        self.features = nn.Sequential(

            # -------------------- BLOCK 1 --------------------
            # Input: (3 x 32 x 32)
            nn.Conv2d(3, 64, kernel_size=3, padding=1),   # Conv layer 1: output (64 x 32 x 32)
            act,                                          # Applying non-linearity
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Conv layer 2: output (64 x 32 x 32)
            act,
            nn.MaxPool2d(2, 2),                           # Downsample → (64 x 16 x 16)

            # -------------------- BLOCK 2 --------------------
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Conv layer 3: output (128 x 16 x 16)
            act,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# Conv layer 4: output (128 x 16 x 16)
            act,
            nn.MaxPool2d(2, 2),                           # Downsample → (128 x 8 x 8)

            # -------------------- BLOCK 3 --------------------
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# Conv layer 5: output (256 x 8 x 8)
            act,
            nn.MaxPool2d(2, 2),                           # Downsample → (256 x 4 x 4)
        )

        # ---------------------------------------------------------------------
        # CLASSIFIER
        # ---------------------------------------------------------------------
        # It takes the flattened feature maps and classifies them
        # into one of the 10 output classes.
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # Flatten (256x4x4) → 4096
            nn.Linear(256 * 4 * 4, 512),                  # Fully connected layer: 4096 → 512
            act,                                          # Activation for non-linearity
            nn.Dropout(0.5),                              # Randomly drop 50% neurons during training
            nn.Linear(512, num_classes),                  # Output layer: 512 → 10 classes
            # No softmax here because CrossEntropyLoss expects raw logits
        )

    # -------------------------------------------------------------------------
    # HELPER FUNCTION — Activation Selector
    # -------------------------------------------------------------------------
    def _get_activation(self, name):
        """
        Returns an activation function layer based on user input.
        Supported activations:
            'relu', 'sigmoid', 'tanh', 'silu'/'swish', 'gelu'
        Default: ReLU
        """
        name = name.lower()
        if name == 'relu': return nn.ReLU(inplace=True)
        if name == 'sigmoid': return nn.Sigmoid()
        if name == 'tanh': return nn.Tanh()
        if name in ('silu', 'swish'): return nn.SiLU()
        if name == 'gelu': return nn.GELU()
        return nn.ReLU(inplace=True)  # Default activation function if name is invalid

    # -------------------------------------------------------------------------
    # FORWARD PASS
    # -------------------------------------------------------------------------
    def forward(self, x):
        """
        Defines how data moves through the network (forward propagation).

        Args:
            x (Tensor): Input image batch, shape = (N, 3, 32, 32)

        Returns:
            Tensor of shape (N, num_classes) — raw class scores (logits)
        """
        # Step 1: Extract hierarchical features using convolutional blocks
        x = self.features(x)

        # Step 2: Flatten and pass through fully connected layers
        x = self.classifier(x)

        # Step 3: Output logits (before softmax)
        return x
