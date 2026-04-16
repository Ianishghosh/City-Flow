import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """Spatial component — learns relationships between road sensors."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x   : (batch, sensors, features)
        # adj : (sensors, sensors)
        support = self.weight(x)
        adj_exp = adj.unsqueeze(0).expand(x.size(0), -1, -1)
        output  = torch.bmm(adj_exp, support)
        return F.relu(output)


class TemporalConvolution(nn.Module):
    """Temporal component — learns patterns across time steps."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x : (batch, time, features)
        x = x.permute(0, 2, 1)   # → (batch, features, time)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x.permute(0, 2, 1)  # → (batch, time, features)


class STGNNBlock(nn.Module):
    """One Spatio-Temporal block: Temporal → Spatial."""

    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.temporal = TemporalConvolution(in_features, hidden_features)
        self.spatial  = GraphConvolution(hidden_features, hidden_features)
        self.dropout  = nn.Dropout(0.3)

    def forward(self, x, adj):
        x = self.temporal(x)
        x = x + self.dropout(x)   # residual
        return x


class STGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for traffic speed prediction.

    Args:
        num_sensors  : number of road sensors (207 for METR-LA)
        input_steps  : historical timesteps used as input  (12 = 1 hour)
        output_steps : future  timesteps to predict        (12 = 1 hour)
        hidden       : hidden dimension size
    """

    def __init__(self, num_sensors=207, input_steps=12, output_steps=12, hidden=64):
        super().__init__()

        self.input_steps  = input_steps
        self.output_steps = output_steps
        self.num_sensors  = num_sensors

        # Input projection
        self.input_proj = nn.Linear(num_sensors, hidden)

        # ST blocks
        self.st_block1 = STGNNBlock(hidden, hidden)
        self.st_block2 = STGNNBlock(hidden, hidden)

        # GRU for temporal sequence modelling
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_sensors * output_steps)
        )

    def forward(self, x, adj):
        # x   : (batch, input_steps, num_sensors)
        # adj : (num_sensors, num_sensors)
        batch = x.size(0)

        x = self.input_proj(x)        # (batch, time, hidden)
        x = self.st_block1(x, adj)
        x = self.st_block2(x, adj)

        x, _ = self.gru(x)            # (batch, time, hidden)
        x    = x[:, -1, :]            # last timestep → (batch, hidden)

        x = self.output_proj(x)       # (batch, sensors * output_steps)
        x = x.view(batch, self.output_steps, self.num_sensors)
        return x


def build_model(cfg):
    """Build model from Config object."""
    return STGNN(
        num_sensors  = cfg.num_sensors,
        input_steps  = cfg.input_steps,
        output_steps = cfg.output_steps,
        hidden       = cfg.hidden_dim
    )