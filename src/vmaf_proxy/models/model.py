import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def safe_gn(groups, ch):
    g = min(groups, ch)
    if ch % g != 0:
        g = math.gcd(ch, g) or 1
    return nn.GroupNorm(g, ch)

def conv_3d(in_channels, out_channels, kernel_size, activation, dropout, norm_groups):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False),
        safe_gn(norm_groups, out_channels),
        activation(),
        nn.Dropout3d(p=dropout),
        nn.MaxPool3d(kernel_size=(1, 2, 2))
    )

class SEBlock3d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        b, c, d, h, w = x.size()
        y = F.adaptive_avg_pool3d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1, 1)
        return x * y


class VMAFNet(nn.Module):
    def __init__(
        self,
        width=1.0,
        dropout=0.1,
        activation='relu',
        norm_groups=32,
        kernel_size=3,
        reduction=16,
        num_conv_layers=7
    ):
        super().__init__()

        # Convert activation str to module
        if activation == 'relu':
            act_cls = nn.ReLU
        elif activation == 'leaky_relu':
            act_cls = lambda: nn.LeakyReLU(0.01)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        C = lambda x: int(x * width)
        self.embed_channels = C(512)

        # Dynamic number of conv layers
        conv_out_channels = [C(32), C(64), C(96), C(128), C(160), C(192), C(256)]
        layers = []
        in_ch = 1
        for i in range(min(num_conv_layers, len(conv_out_channels))):
            layers.append(conv_3d(in_ch, conv_out_channels[i], kernel_size, act_cls, dropout, norm_groups))
            in_ch = conv_out_channels[i]
        self.cnn1 = nn.Sequential(*layers)

        self.cnn2 = nn.Sequential(
            nn.Conv3d(in_ch, self.embed_channels, kernel_size, padding=kernel_size//2, bias=False),
            safe_gn(norm_groups, self.embed_channels),
            act_cls(),
            nn.Dropout3d(p=dropout),
        )

        self.cnn3 = nn.Sequential(
            nn.Conv3d(
                self.embed_channels * 3,
                self.embed_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2,
                bias=False
            ),
            safe_gn(norm_groups, self.embed_channels),
            act_cls(),
            nn.Dropout3d(p=dropout),
            SEBlock3d(self.embed_channels, reduction=reduction),
        )

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.embed_channels, 128),
            act_cls(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_ref, x_dist):
        x_ref = x_ref.unsqueeze(1)  # B x 1 x N x H x W
        x_dist = x_dist.unsqueeze(1)

        f_ref = self.cnn1(x_ref)
        f_dist = self.cnn1(x_dist)

        f_ref = self.cnn2(f_ref)
        f_dist = self.cnn2(f_dist)

        #Helps for quality deltas
        x = torch.cat([f_ref, f_dist, torch.abs(f_ref - f_dist)], dim=1)

        x = self.cnn3(x)
        x = self.global_pool(x)
        x = self.regressor(x).squeeze(-1)

        return x