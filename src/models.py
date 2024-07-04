import math

import torch
from loguru import logger
from torch import Tensor, nn
import torch.nn.functional as F


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout):
#         super().__init__()
#   #      dropout = config['dropout']
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#    #         nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(dropout)
#         )

#         # Define a 1x1 convolution to match the dimensions if necessary
#         self.match_dimensions = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()
        
#         # BatchNorm layer after the addition of skip connection
#         self.final_norm = nn.BatchNorm2d(out_channels)

#     def forward(self, x):
#         identity = x.clone() # Save the input for the skip connection
#         x = self.conv(x) # Pass through the convolutional block
#         identity = self.match_dimensions(identity) # Match dimensions if necessary
#         x += identity # Add the original input (skip connection)
#         x = self.final_norm(x) # Normalize the output
#         return x


# class CNN(nn.Module):
#     def __init__(self, config: dict) -> None:
#         super().__init__()
#         hidden = config['hidden']
#         dropout = config['dropout']
#         self.convolutions = nn.ModuleList([
#             ConvBlock(1, hidden, dropout),
#         ])


#         for i in range(config['num_layers']):
#             self.convolutions.extend([ConvBlock(hidden, hidden, dropout), nn.ReLU()])
#         self.convolutions.append(nn.MaxPool2d(2, 2))

#         self.dense = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear((8*6) * hidden, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),  # Add dropout here
#             nn.Linear(hidden, config['num_classes']),
#             nn.Sigmoid()
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for conv in self.convolutions:
#             x = conv(x)
# #            print(f'After {conv.__class__.__name__}, shape: {x.shape}')
#         x = self.dense(x)
#         return x


class ConvBlock_RG(nn.Module):
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


class CNN_RG(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden = config["hidden"]
        dropout = config['dropout']
        self.convolutions = nn.ModuleList(
            [
                ConvBlock(1, hidden),
            ]
        )

        for i in range(config["num_layers"]):
            self.convolutions.extend([ConvBlock(hidden, hidden)])
        self.convolutions.append(nn.MaxPool2d(2, 2))

        activation_map_size = config["shape"][0] // 2 * config["shape"][1] // 2
        logger.info(f"Activation map size: {activation_map_size}")
        logger.info(f"Input linear: {activation_map_size * hidden}")

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(activation_map_size * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),  # Add dropout here
            nn.Linear(hidden, config["num_classes"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convolutions:
            x = conv(x)
        x = self.dense(x)
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        dropout = config['dropout']
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.conv(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class CNN(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden = config['hidden']
        dropout = config['dropout']
        self.convolutions = nn.ModuleList([
            ConvBlock(1, hidden),
        ])

#        for i in range(config['num_layers']):
#            self.convolutions.extend([ConvBlock(hidden, hidden), nn.ReLU()])

        for i in range(config['num_layers']):
            self.convolutions.extend([ResidualBlock(hidden, hidden), nn.ReLU()])
        
        self.convolutions.append(nn.MaxPool2d(2, 2))

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear((8*6) * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),  # Add dropout here
            nn.Linear(hidden, config['num_classes']),
 #           nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convolutions:
            x = conv(x)
#            print(f'After {conv.__class__.__name__}, shape: {x.shape}')
        x = self.dense(x)
        return x
    
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)
    
    def forward(self, outputs, targets):
        return self.criterion(outputs, targets)


class PositionalEncoding(nn.Module):
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
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
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
