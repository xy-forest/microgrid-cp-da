"""Forecasting model implementations: LSTM, Transformer, and PatchTST.

Each model takes a sequence of shape (batch_size, seq_len, n_features) and
outputs a scalar prediction for the target variable at horizon H.
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """Bidirectional LSTM with a two-layer MLP prediction head.

    Parameters
    ----------
    n_features : int
        Number of input features per time step.
    hidden_size : int
        LSTM hidden state dimension (doubled for bidirectional output).
    n_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout rate applied between LSTM layers.
    """

    def __init__(
        self,
        n_features: int = 3,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar prediction from the last LSTM time step.

        Parameters
        ----------
        x : Tensor of shape (B, T, F)
            Input sequence.

        Returns
        -------
        Tensor of shape (B, 1)
            Predicted value at the forecast horizon.
        """
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class TransformerForecaster(nn.Module):
    """Transformer encoder adapted for time series forecasting.

    Uses nn.TransformerEncoder with learned input projection; no explicit
    positional encoding is added (the 60-step context is short enough that
    the attention mechanism can learn temporal order from the feature patterns
    alone).

    Parameters
    ----------
    n_features : int
        Number of input features.
    d_model : int
        Embedding dimension.
    nhead : int
        Number of attention heads.
    n_layers : int
        Number of transformer encoder layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_features: int = 3,
        d_model: int = 48,
        nhead: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar prediction from the last encoder position.

        Parameters
        ----------
        x : Tensor of shape (B, T, F)

        Returns
        -------
        Tensor of shape (B, 1)
        """
        x = self.input_proj(x)
        out = self.encoder(x)
        return self.head(out[:, -1, :])


class PatchTSTForecaster(nn.Module):
    """Patch-based time series transformer (Nie et al., 2023).

    Divides the input sequence into overlapping patches, projects each patch
    into a d_model embedding, then applies a transformer encoder.

    Parameters
    ----------
    n_features : int
        Number of input features.
    d_model : int
        Embedding dimension per patch.
    nhead : int
        Number of attention heads.
    n_layers : int
        Number of encoder layers.
    patch_len : int
        Length of each patch.
    stride : int
        Stride between consecutive patches.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_features: int = 3,
        d_model: int = 48,
        nhead: int = 4,
        n_layers: int = 2,
        patch_len: int = 6,
        stride: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.input_proj = nn.Linear(patch_len * n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patches, encode, and predict.

        Parameters
        ----------
        x : Tensor of shape (B, T, F)

        Returns
        -------
        Tensor of shape (B, 1)
        """
        # x.unfold creates overlapping patches in the time dimension
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, P, F, patch_len)
        B, P, F, L = patches.shape
        patches = patches.permute(0, 1, 3, 2).reshape(B, P, L * F)  # (B, P, F*L)
        patches = self.input_proj(patches)  # (B, P, d_model)
        out = self.encoder(patches)
        return self.head(out[:, -1, :])
