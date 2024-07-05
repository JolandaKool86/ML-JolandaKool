import math

import torch
from loguru import logger
from torch import Tensor, nn


class ConvBlock_BM(nn.Module):
    """
    Convolutional Block with Batch Normalization and ReLU activation.

    This block consists of two convolutional layers, each followed by a ReLU activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Attributes:
        conv (nn.Sequential): Sequential container with two convolutional layers and ReLU activations.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class CNN_BM(nn.Module):
    """
    Convolutional Neural Network with multiple convolutional blocks and dense layers.

    Args:
        config (dict): Configuration dictionary containing:
            - hidden (int): Number of hidden units.
            - dropout (float): Dropout rate.
            - num_layers (int): Number of convolutional layers.
            - shape (tuple): Shape of the input data (height, width).
            - num_classes (int): Number of output classes.

    Attributes:
        convolutions (nn.ModuleList): List of convolutional and pooling layers.
        dense (nn.Sequential): Fully connected layers for classification.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden = config["hidden"]
        dropout = config["dropout"]
        self.convolutions = nn.ModuleList(
            [
                ConvBlock_BM(1, hidden, dropout),
            ]
        )

        for i in range(config["num_layers"]):
            self.convolutions.extend([ConvBlock_BM(hidden, hidden)])
        self.convolutions.append(nn.MaxPool2d(2, 2))

        activation_map_size = config["shape"][0] // 2 * config["shape"][1] // 2
        logger.info(f"Activation map size: {activation_map_size}")
        logger.info(f"Input linear: {activation_map_size * hidden}")

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(activation_map_size * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config["num_classes"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convolutions:
            x = conv(x)
        x = self.dense(x)
        return x


class ResBlock(nn.Module):
    """
    Residual Block with Convolutional Layers and Batch Normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dropout (float): Dropout rate.

    Attributes:
        conv (nn.Sequential): Sequence of convolutional, batch normalization, ReLU, and dropout layers.
        match_dimensions (nn.Module): Convolutional layer or identity mapping to match input and output dimensions.
        final_norm (nn.BatchNorm2d): Batch normalization layer applied after adding the identity shortcut.
    """

    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.match_dimensions = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.final_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x.clone()
        x = self.conv(x)
        identity = self.match_dimensions(identity)
        x += identity
        x = self.final_norm(x)
        return x


class CNN(nn.Module):
    """
    Convolutional Neural Network with Residual and Convolutional Blocks.

    Args:
        config (dict): Configuration dictionary containing hidden units, dropout rate, number of layers, and number of output classes.

    Attributes:
        convolutions (nn.ModuleList): List of convolutional and residual blocks.
        dense (nn.Sequential): Fully connected layers for classification.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden = config["hidden"]
        dropout = config["dropout"]
        self.convolutions = nn.ModuleList(
            [
                ResBlock(1, hidden, dropout),
            ]
        )

        for i in range(config["num_layers"]):
            self.convolutions.extend([ResBlock(hidden, hidden, dropout), nn.ReLU()])
        self.convolutions.append(nn.MaxPool2d(2, 2))

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear((8 * 6) * hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),  # Add dropout here
            nn.Linear(hidden, config["num_classes"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convolutions:
            x = conv(x)
        x = self.dense(x)
        return x


class WeightedCrossEntropyLoss(nn.Module):
    """
    Custom implementation of CrossEntropyLoss with weights for imbalanced classes.

    Args:
        weight (Tensor): A manual rescaling weight given to each class. If given, it has to be a Tensor of size `C`.

    Attributes:
        weight (Tensor): The weight tensor.
        criterion (nn.CrossEntropyLoss): The weighted cross-entropy loss function.
    """

    def __init__(self, weight):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)

    def forward(self, outputs, targets):
        return self.criterion(outputs, targets)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for adding positional information to the input embeddings.

    Args:
        d_model (int): The dimension of the model.
        dropout (float): The dropout rate. Default is 0.1.
        max_seq_len (int): The maximum sequence length. Default is 5000.

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        pe (Tensor): Positional encoding matrix.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_seq_len, d_model)
        # batch, seq_len, d_model
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Transformer Block consisting of multi-head self-attention and feed-forward layers.

    Args:
        hidden_size (int): The size of the hidden layers.
        num_heads (int): The number of attention heads.
        dropout (float): The dropout rate.

    Attributes:
        attention (nn.MultiheadAttention): Multi-head self-attention layer.
        ff (nn.Sequential): Feed-forward neural network.
        layer_norm1 (nn.LayerNorm): Layer normalization after attention layer.
        layer_norm2 (nn.LayerNorm): Layer normalization after feed-forward layer.
    """

    def __init__(self, hidden_size, num_heads, dropout):
        # feel free to change the input parameters of the constructor
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        identity = x.clone()  # skip connection
        x, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + identity)  # Add & Norm skip
        identity = x.clone()  # second skip connection
        x = self.ff(x)
        x = self.layer_norm2(x + identity)  # Add & Norm skip
        return x


class Transformer(nn.Module):
    """
    Transformer model consisting of convolutional input processing, positional encoding, and multiple transformer blocks.

    Args:
        config (dict): Configuration dictionary containing:
            - hidden (int): Number of hidden units.
            - dropout (float): Dropout rate.
            - num_heads (int): Number of attention heads.
            - num_blocks (int): Number of transformer blocks.
            - output (int): Number of output classes.

    Attributes:
        conv1d (nn.Conv1d): 1D convolutional layer for initial input processing.
        pos_encoder (PositionalEncoding): Positional encoding module.
        transformer_blocks (nn.ModuleList): List of transformer blocks.
        out (nn.Linear): Linear layer for final output.
    """

    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=config["hidden"],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.pos_encoder = PositionalEncoding(config["hidden"], config["dropout"])

        # Create multiple transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config["hidden"], config["num_heads"], config["dropout"]
                )
                for _ in range(config["num_blocks"])
            ]
        )

        self.out = nn.Linear(config["hidden"], config["output"])

    def forward(self, x: Tensor) -> Tensor:
        # streamer:         (batch, seq_len, channels)
        # conv1d:           (batch, channels, seq_len)
        # pos_encoding:     (batch, seq_len, channels)
        # attention:        (batch, seq_len, channels)
        x = self.conv1d(x.transpose(1, 2))  # flip channels and seq_len for conv1d
        x = self.pos_encoder(x.transpose(1, 2))  # flip back to seq_len and channels

        # Apply multiple transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = x.mean(dim=1)  # Global Average Pooling
        x = self.out(x)
        return x
