import torch.nn as nn
from models.focalnet import FocalNet
from collections import OrderedDict
import torch


class WeakFocalNet3D(nn.Module):
    def __init__(self, img_size, patch_size, in_ch, num_classes, embed_dims, depths, levels, windows, use_conv_embed=True):
        super().__init__()
        self.num_layers = len(depths)
        features_dims = embed_dims * 2 ** (self.num_layers - 1)

        self.focal_encoder = FocalNet(img_size=img_size, patch_size=patch_size, in_chans=in_ch, embed_dim=embed_dims, depths=depths, num_classes=num_classes,
                                      focal_levels=levels, focal_windows=windows, use_conv_embed=use_conv_embed)

        self.modulators_conv_blocks = nn.ModuleDict()
        self.modulators_conv_blocks['input_conv'] = nn.Sequential(nn.Conv3d(in_ch, embed_dims, kernel_size=3, padding='same'), nn.GELU(), nn.MaxPool3d(kernel_size=2, stride=2))
        for i in range(self.num_layers):
            in_chans = embed_dims * 2 ** i
            out_chans = in_chans * 2
            self.modulators_conv_blocks[f'layer{i}-conv'] = nn.Sequential(nn.Conv3d(in_chans * 2, out_chans, kernel_size=3, padding='same'),
                                                                          nn.GELU(),
                                                                          nn.Conv3d(out_chans, out_chans, kernel_size=3, padding='same'),
                                                                          nn.GELU())
            self.modulators_conv_blocks[f'layer{i}-pool'] = nn.MaxPool3d(kernel_size=2, stride=2)

        self.modulators_conv_blocks['bottleneck'] = nn.Sequential(nn.Conv3d(features_dims * 2, features_dims, kernel_size=3, padding='same'),
                                                                  nn.GELU(),
                                                                  nn.AdaptiveAvgPool3d(1),
                                                                  nn.Flatten(1),
                                                                  nn.LayerNorm(features_dims))
        self.head = nn.Linear(features_dims, num_classes)

        self.up_sample = nn.Upsample((img_size, img_size, img_size), mode='trilinear')

    def forward(self, x):
        focal_features = self.focal_encoder.forward_features(x)

        x = self.modulators_conv_blocks['input_conv'](x)
        for i in range(self.num_layers):
            x_modulator_concat = torch.cat((x, self.focal_encoder.layers[i].blocks[-1].modulation.modulator), dim=1)
            x = self.modulators_conv_blocks[f'layer{i}-conv'](x_modulator_concat) + x_modulator_concat
            x = self.modulators_conv_blocks[f'layer{i}-pool'](x)
        x = self.modulators_conv_blocks['bottleneck'](x)

        return self.head(focal_features * x)

    def segmentation(self, x):
        y = self.forward(x)

        mask = torch.abs(self.focal_encoder.layers[-1].blocks[-1].modulation.modulator).mean(1, keepdim=True)
        mask = self.up_sample(mask)

        return y, mask.squeeze()
