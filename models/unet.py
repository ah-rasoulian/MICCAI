import torch.nn as nn
from collections import OrderedDict
import torch


class UNet(nn.Module):
    def __init__(self, in_ch, num_classes, embed_dims):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims

        encoder = OrderedDict()
        for i in range(len(self.embed_dims) - 1):
            encoder[f'contracting-{i + 1}'] = ResConvBlock(in_ch=in_ch if i == 0 else self.embed_dims[i - 1], out_ch=self.embed_dims[i], kernel_size=3)
            encoder[f'pool-{i + 1}'] = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder = nn.ModuleDict(encoder)

        self.bottleneck = ResConvBlock(in_ch=self.embed_dims[-2], out_ch=self.embed_dims[-1], kernel_size=3)

        self.embed_dims.reverse()
        self.decoder = nn.ModuleDict()
        for i in range(len(self.embed_dims) - 1):
            self.decoder[f'up-{i + 1}'] = nn.ConvTranspose3d(in_channels=self.embed_dims[i], out_channels=self.embed_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1)
            self.decoder[f'expansive-{i + 1}'] = ResConvBlock(in_ch=2 * self.embed_dims[i + 1], out_ch=self.embed_dims[i + 1], kernel_size=3,
                                                              drop_rate=0.25)

        self.segmentation_head = nn.Conv3d(in_channels=self.embed_dims[-1], out_channels=num_classes, kernel_size=3, padding='same')

    def forward(self, x):
        residuals = []
        for i in range(len(self.embed_dims) - 1):
            x = self.encoder[f'contracting-{i + 1}'](x)
            residuals.append(x)
            x = self.encoder[f'pool-{i + 1}'](x)

        x = self.bottleneck(x)
        residuals.reverse()
        for i in range(len(self.embed_dims) - 1):
            x = self.decoder[f'up-{i + 1}'](x)
            concat = torch.cat((x, residuals[i]), dim=1)
            x = self.decoder[f'expansive-{i + 1}'](concat)

        return self.segmentation_head(x)


class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, drop_rate=0.):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size, padding='same')
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size, padding='same')
        self.norm2 = nn.InstanceNorm3d(out_ch)

        self.conv3 = nn.Conv3d(in_ch, out_ch, kernel_size, padding='same')
        self.norm3 = nn.InstanceNorm3d(out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout3d(drop_rate)

    def forward(self, x):
        residual = x
        x = self.conv1(x)  # b, c, d, h, w
        x = self.norm1(x)  # b, d, h, w, c
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)

        residual = self.conv3(residual)
        residual = self.norm3(residual)

        x = x + residual
        x = self.act(x)
        return self.dropout(x)
