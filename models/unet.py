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
            encoder[f'contracting-{i + 1}'] = ConvBlock(in_ch=in_ch if i == 0 else self.embed_dims[i - 1], out_ch=self.embed_dims[i], kernel_size=3)
            encoder[f'pool-{i + 1}'] = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder = nn.ModuleDict(encoder)

        self.bottleneck = ConvBlock(in_ch=self.embed_dims[-2], out_ch=self.embed_dims[-1], kernel_size=3)

        self.embed_dims.reverse()
        self.decoder = nn.ModuleDict()
        for i in range(len(self.embed_dims) - 1):
            self.decoder[f'up-{i + 1}'] = nn.ConvTranspose3d(in_channels=self.embed_dims[i], out_channels=self.embed_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1)
            self.decoder[f'dropout-{i}'] = nn.Dropout3d(p=0.25)
            self.decoder[f'expansive-{i + 1}'] = ConvBlock(in_ch=2 * self.embed_dims[i + 1], out_ch=self.embed_dims[i + 1], kernel_size=3)

        self.segmentation_head = nn.Sequential(nn.Conv3d(in_channels=self.embed_dims[-1], out_channels=num_classes, kernel_size=1, padding='same'),
                                               nn.Softmax(dim=1))

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
            concat = self.decoder[f'dropout-{i}'](concat)
            x = self.decoder[f'expansive-{i + 1}'](concat)

        return self.segmentation_head(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, droprate=0.0):
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size, padding='same'),
                                   nn.ReLU(),
                                   nn.Conv3d(out_ch, out_ch, kernel_size, padding='same'),
                                   nn.ReLU(),
                                   nn.BatchNorm3d(num_features=out_ch),
                                   nn.Dropout3d(droprate)
                                   )

    def forward(self, x):
        return self.block(x)
