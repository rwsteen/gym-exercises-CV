import torch
import torch.nn as nn

from .graph import PennActionGraph
from .gcn import unit_gcn
from .tcn import unit_tcn

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A,
                 stride=1):

        super().__init__()

        # Spatial Graph Convolution (adaptive)
        self.gcn = unit_gcn(
            in_channels,
            out_channels,
            A,
            with_res=True
        )

        # Temporal Convolution
        self.tcn = unit_tcn(
            out_channels,
            out_channels,
            kernel_size=9,
            stride=stride
        )

        self.relu = nn.ReLU()

        # Residual
        self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        x = self.tcn(self.gcn(x)) + res
        return self.relu(x)
    

class STGCN(nn.Module):

    def __init__(self,
                 num_class,
                 num_point,
                 num_person=1,
                 in_channels=2):

        super().__init__()

        self.graph = PennActionGraph(strategy='spatial')
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # backbone layers
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, A),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A, stride=2),
            STGCNBlock(128, 128, A),
            STGCNBlock(128, 256, A, stride=2),
            STGCNBlock(256, 256, A),
        ])

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # Head 1: action classification
        self.action_head = nn.Linear(256, num_class)

        # Head 2: phase prediction
        self.phase_head = nn.Conv1d(256, 1, kernel_size=1)

    def forward(self, x):
        """
        x: (N, C, T, V, M) - batch size, channels, time steps, joints, persons
        """

        N, C, T, V, M = x.shape

        # reshape for BN
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (N, M, C, T, V)
        x = self.data_bn(x.view(N, M*C*V, T))
        x = x.view(N, M, C, V, T).permute(0, 2, 4, 3, 1) # (N, C, T, V, M)
        x = x.contiguous().view(N*M, C, T, V)

        # ST-GCN layers
        for layer in self.layers:
            x = layer(x)

        # reshape back
        x = x.view(N, M, x.size(1), x.size(2), x.size(3)) # (N, M, C_out, T_out, V_out)

        # average over persons and joints
        x = x.mean(dim=[1,4])   # (N, C_out, T_out)

        # action classification
        action_feat = x.mean(dim=2)  # (N, C_out)
        action = self.action_head(action_feat) # (N, num_class)

        # phase prediction
        phase = self.phase_head(x)   # (N, 1, T_out)
        phase = phase.squeeze(1)    # (N, T_out)

        return action, phase