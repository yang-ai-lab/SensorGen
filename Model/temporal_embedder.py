"""
Temporal feature extractors for time series conditioning.
Used for forecasting tasks with past/context signals.
"""

import torch
import torch.nn as nn


class TemporalFeatureExtractor(nn.Module):
    """
    Extract features from time series for dense conditioning.
    Uses 1D convolutions to embed temporal patterns.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        context_length: int,
        patch_size: int = 10,
        depth: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.context_length = context_length
        self.patch_size = patch_size
        self.num_patches = context_length // patch_size
        
        # Patchify: conv1d with stride=patch_size
        self.patch_embed = nn.Conv1d(
            in_channels, hidden_size, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Simple transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T] time series (e.g., past context)
        Returns:
            [B, L, D] dense embeddings for cross-attention
        """
        B, C, T = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        assert T == self.context_length, f"Expected length {self.context_length}, got {T}"
        
        # Patchify
        x = self.patch_embed(x)  # [B, D, L]
        x = x.transpose(1, 2)  # [B, L, D]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Encode temporal patterns
        x = self.encoder(x)  # [B, L, D]
        
        return x


class SimpleTemporalPooler(nn.Module):
    """
    Feature extraction + pooling for sparse conditioning from time series.
    Uses 1D convolutions to extract features before pooling.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        series_length: int,
        pool_type: str = 'mean',
        num_conv_layers: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.series_length = series_length
        self.pool_type = pool_type
        
        # Multi-scale feature extraction with 1D convolutions
        # Use multiple kernel sizes to capture different temporal patterns
        self.conv_layers = nn.ModuleList()
        
        # First conv: expand channels
        self.conv_layers.append(nn.Sequential(
            nn.Conv1d(in_channels, hidden_size // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
        ))
        
        # Middle convs: refine features
        for _ in range(num_conv_layers - 2):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(hidden_size // 2, hidden_size // 2, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_size // 2),
                nn.GELU(),
            ))
        
        # Final conv: map to hidden_size
        self.conv_layers.append(nn.Sequential(
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
        ))
        
        # Optional: learnable pooling weights
        if pool_type == 'attention':
            self.pool_attn = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T] time series
        Returns:
            [B, D] sparse embedding for AdaLN
        """
        B, C, T = x.shape
        
        # Feature extraction through conv layers
        for conv in self.conv_layers:
            x = conv(x)  # [B, hidden_size, T]
        
        # Pool over time
        if self.pool_type == 'mean':
            x = x.mean(dim=2)  # [B, D]
        elif self.pool_type == 'max':
            x = x.max(dim=2)[0]  # [B, D]
        elif self.pool_type == 'attention':
            # Attention-based pooling
            x_t = x.transpose(1, 2)  # [B, T, D]
            attn_weights = self.pool_attn(x_t).squeeze(-1)  # [B, T]
            attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(1)  # [B, 1, T]
            x = (attn_weights @ x_t).squeeze(1)  # [B, 1, T] @ [B, T, D] = [B, 1, D] → [B, D]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        return x


class TemporalBoundaryFeatureExtractor(TemporalFeatureExtractor):
    """
    Specialized feature extractor for temporal boundary imputation.
    Inherits from TemporalFeatureExtractor and adds boundary markers
    to distinguish past vs. future boundaries.
    
    This is a separate class to maintain backward compatibility with
    existing checkpoints that use the base TemporalFeatureExtractor.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        context_length: int,
        patch_size: int = 10,
        depth: int = 2,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_size=hidden_size,
            context_length=context_length,
            patch_size=patch_size,
            depth=depth
        )
        
        # Add boundary markers to distinguish past vs. future
        self.boundary_markers = nn.Parameter(
            torch.zeros(2, 1, hidden_size)  # [past_marker, future_marker]
        )
        nn.init.normal_(self.boundary_markers, std=0.02)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T] time series - concatenated left+right boundaries
               First half (0:T/2) is past boundary, second half (T/2:T) is future boundary
        Returns:
            [B, L, D] dense embeddings with boundary markers applied
        """
        B, C, T = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        assert T == self.context_length, f"Expected length {self.context_length}, got {T}"
        
        # Split into left and right boundaries
        T_half = T // 2
        x_left = x[:, :, :T_half]   # (B, C, T/2) - Past boundary
        x_right = x[:, :, T_half:]  # (B, C, T/2) - Future boundary
        
        # Patchify separately
        x_left = self.patch_embed(x_left)   # (B, D, L/2)
        x_right = self.patch_embed(x_right) # (B, D, L/2)
        
        x_left = x_left.transpose(1, 2)   # (B, L/2, D)
        x_right = x_right.transpose(1, 2) # (B, L/2, D)
        
        L_half = x_left.shape[1]
        
        # Add positional embeddings (same positions for both halves)
        x_left = x_left + self.pos_embed[:, :L_half, :]
        x_right = x_right + self.pos_embed[:, :L_half, :]
        
        # Add boundary markers to distinguish past vs. future
        x_left = x_left + self.boundary_markers[0:1, :, :]   # Past marker
        x_right = x_right + self.boundary_markers[1:2, :, :]  # Future marker
        
        # Concatenate marked boundaries
        x = torch.cat([x_left, x_right], dim=1)  # (B, L, D)
        
        # Encode temporal patterns
        x = self.encoder(x)  # (B, L, D)
        
        return x

