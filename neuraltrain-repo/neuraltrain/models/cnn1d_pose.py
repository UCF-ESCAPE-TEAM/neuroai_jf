import torch
from torch import nn
import torch.nn.functional as F

from neuraltrain.models.base import BaseModelConfig


class Cnn1dPose(BaseModelConfig):
    """Config for a small 1D CNN that maps EMG to joint-angle trajectories."""

    hidden: int = 128
    dropout: float = 0.1

    def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:
        return Cnn1dPoseModule(n_in_channels, n_outputs, config=self)


def _block(c_in, c_out, kernel, stride, dropout):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class Cnn1dPoseModule(nn.Module):
    def __init__(self, n_in_channels, n_outputs, config):
        super().__init__()
        h, p = config.hidden, config.dropout
        self.input_norm = nn.BatchNorm1d(n_in_channels)   # raw EMG arrives in microvolts
        self.encoder = nn.Sequential(                     # 2 kHz -> 50 Hz
            _block(n_in_channels, h, 11, 5, p),           # 400 Hz
            _block(h, h, 9, 4, p),                        # 100 Hz
            _block(h, h, 9, 2, p),                        # 50 Hz
        )
        self.body = nn.Sequential(                        # temporal context at 50 Hz
            _block(h, h, 9, 1, p),
            _block(h, h, 9, 1, p),
        )
        self.head = nn.Conv1d(h, n_outputs, kernel_size=1)

    def forward(self, x):
        # x: (batch, 16, samples) -> (batch, 20, samples)
        n_samples = x.shape[-1]
        z = self.body(self.encoder(self.input_norm(x)))
        y = self.head(z)                                  # (batch, 20, samples / 40)
        return F.interpolate(y, size=n_samples, mode="linear", align_corners=False)