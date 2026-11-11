import torch
import torch.nn as nn
import torch.nn.functional as F


class SalientSampleDetector(nn.Module):
    def __init__(self, in_channels, base_channels=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),  # [B,C=base_channels,H,W]
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1),  # 下采样[B,2C,H/2,W/2]
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1),  # [B,4C,H/4,W/4]
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),  # [B,2C,H/2,W/2]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),  # [B,C,H,W]
            nn.ReLU(inplace=True),
        )

        self.heatmap_head = nn.Conv2d(base_channels, 1, 1)

        self.feature_head = nn.Conv2d(base_channels, 128, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        target_size = x.shape[-2:]
        feat = self.encoder(x)  # 输入[B,C,H,W],输出[B,4C,H/4,W/4]
        feat = self.decoder(feat)  # 输入[B,4C,H/4,W/4]，输出[B,C,H,W]

        if feat.shape[-2:] != target_size:
            feat = F.interpolate(
                feat,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        heatmap = torch.sigmoid(self.heatmap_head(feat))  # 概率图 [B,1,H,W]
        heatmap = heatmap.squeeze(0)
        feature_map = F.normalize(self.feature_head(feat), p=2, dim=1)

        return heatmap