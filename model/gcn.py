import torch
import torch.nn as nn

EPS = 1e-4


class unit_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 adaptive='importance',
                 with_res=True):

        super().__init__()

        self.num_subsets = A.size(0)
        self.adaptive = adaptive
        self.with_res = with_res

        # Base adjacency
        self.A = nn.Parameter(A.clone(), requires_grad=(adaptive=='init'))

        # Importance / offset learning
        if adaptive in ['importance', 'offset']:
            self.PA = nn.Parameter(torch.ones_like(A))

        # Feature projection (pre-conv)
        self.conv = nn.Conv2d(in_channels,
                              out_channels * self.num_subsets,
                              kernel_size=1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        # Residual
        if with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

    def forward(self, x):

        n, c, t, v = x.shape
        res = self.down(x)

        # Adaptive adjacency
        if self.adaptive == 'importance':
            A = self.A * self.PA
        elif self.adaptive == 'offset':
            A = self.A + self.PA
        else:
            A = self.A

        # Pre-conv
        x = self.conv(x)
        x = x.view(n, self.num_subsets, -1, t, v)

        # Graph propagation
        x = torch.einsum('nkctv,kvw->nctw', x, A)

        return self.act(self.bn(x) + res)