"""Small 1D CNN that maps wrist EMG to dense joint-angle trajectories."""

import torch.nn.functional as F
from torch import nn

from .base import BaseBrainModelConfig


class Cnn1dPose(BaseBrainModelConfig):
    """Config for a small 1D CNN for EMG-to-pose regression."""

    hidden: int = 128
    dropout: float = 0.1

    # Parameter names must match BrainModelBuildContext fields: the factory
    # injects them by name.
    def build(self, n_spatial_locations: int, n_outputs: int | None) -> nn.Module:
        assert n_outputs is not None, "Cnn1dPose requires n_outputs"
        return Cnn1dPoseModel(
            n_spatial_locations, n_outputs, hidden=self.hidden, dropout=self.dropout
        )


def _block(c_in: int, c_out: int, kernel: int, stride: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class Cnn1dPoseModel(nn.Module):
    """``nn.Module`` implementation of :class:`Cnn1dPose`."""

    def __init__(
        self, n_in_channels: int, n_outputs: int, hidden: int = 128, dropout: float = 0.1
    ):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(n_in_channels)  # raw EMG arrives in microvolts
        self.encoder = nn.Sequential(  # 2 kHz -> 50 Hz
            _block(n_in_channels, hidden, 11, 5, dropout),  # 400 Hz
            _block(hidden, hidden, 9, 4, dropout),  # 100 Hz
            _block(hidden, hidden, 9, 2, dropout),  # 50 Hz
        )
        self.body = nn.Sequential(  # temporal context at 50 Hz
            _block(hidden, hidden, 9, 1, dropout),
            _block(hidden, hidden, 9, 1, dropout),
        )
        self.head = nn.Conv1d(hidden, n_outputs, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, n_outputs), the time-major layout neuralbench expects
        n_samples = x.shape[-1]
        z = self.body(self.encoder(self.input_norm(x)))
        y = self.head(z)  # (B, n_outputs, T / 40)
        y = F.interpolate(y, size=n_samples, mode="linear", align_corners=False)
        return y.transpose(1, 2)