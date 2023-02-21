from models.focalnet import FocalNet, PatchEmbed
import torch.nn as nn
import torch
from timm.models.layers import to_3tuple
from models.unet import ConvBlock
import numpy as np
from collections import OrderedDict


class MultiTaskFocalUnet(nn.Module):
    def __init__(self, img_size, patch_size, in_ch, num_classes, embed_dims, depths, levels, windows, classification_droprate=0.2):
        super().__init__()
        self.num_layers = len(depths)
        features_dims = embed_dims * 2 ** (self.num_layers - 1)

        self.focal_encoder = FocalNet(img_size=img_size, patch_size=patch_size, in_chans=in_ch, embed_dim=embed_dims, depths=depths, num_classes=-1,
                                      focal_levels=levels, focal_windows=windows, use_conv_embed=True)

        self.input_embedder = ConvBlock(in_ch=in_ch, out_ch=embed_dims, kernel_size=3)
        self.bottleneck_patch_merging = PatchEmbed(to_3tuple(img_size // 2 ** self.num_layers), 2,  features_dims, 2 * features_dims, use_conv_embed=True, norm_layer=nn.LayerNorm, is_stem=False)
        self.final_conv_block = ConvBlock(in_ch=2 * embed_dims, out_ch=embed_dims, kernel_size=3)
        self.patch_upsample = nn.ConvTranspose3d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.focal_residuals = OrderedDict()
        self.focal_encoder.layers[0].register_forward_hook(get_intermediate_residuals(self.focal_residuals, f'layer-0'))
        for ln, layer in enumerate(self.focal_encoder.layers):
            layer.register_forward_hook(get_intermediate_residuals(self.focal_residuals, f'layer-{ln + 1}'))

        self.residual_conv_blocks = nn.ModuleDict()
        for i in range(self.num_layers + 1):
            in_chans = embed_dims * 2 ** i
            out_chans = in_chans
            self.residual_conv_blocks[f'layer-{i}'] = ConvBlock(in_ch=in_chans, out_ch=out_chans, kernel_size=3)

        self.decoder = nn.ModuleDict()
        for i in range(1, self.num_layers + 1):
            out_chans = embed_dims * 2 ** (self.num_layers - i)
            in_chans = out_chans * 2
            self.decoder[f'up-{i}'] = nn.ConvTranspose3d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.decoder[f'expansive-{i}'] = ConvBlock(in_ch=2 * out_chans, out_ch=out_chans, kernel_size=3)

        self.segmentation_head = nn.Conv3d(in_channels=embed_dims, out_channels=num_classes, kernel_size=3, padding='same')
        self.classification_head = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                                                 nn.Flatten(1),
                                                 nn.Dropout(classification_droprate),
                                                 nn.Linear(features_dims * 2, num_classes)
                                                 )

    def forward(self, x):
        focal_features = self.focal_encoder.forward_features(x)

        bottleneck, d, h, w = self.bottleneck_patch_merging(self.focal_residuals[f'layer-{self.num_layers}'])
        b, L, c = bottleneck.shape
        bottleneck = bottleneck.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)

        residuals = []
        for i in range(self.num_layers):
            residuals.append(self.residual_conv_blocks[f'layer-{i}'](self.focal_residuals[f'layer-{i}']))

        f = self.residual_conv_blocks[f'layer-{self.num_layers}'](bottleneck)
        for i in range(1, self.num_layers + 1):
            f = self.decoder[f'up-{i}'](f)
            f = self.decoder[f'expansive-{i}'](torch.cat((f, residuals[self.num_layers - i]), dim=1))

        x = self.input_embedder(x)
        x = self.final_conv_block(torch.cat((x, self.patch_upsample(f)), dim=1))
        return self.classification_head(bottleneck), self.segmentation_head(x)


def get_intermediate_residuals(saved_dict, layer_name):
    def hook(model, inp, out):
        if layer_name == 'layer-0':
            residual, d, h, w = inp
        else:
            residual, d, h, w = out
        b, L, c = residual.shape
        assert L == d * h * w
        residual = residual.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)
        saved_dict[layer_name] = residual
    return hook
