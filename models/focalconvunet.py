import torch
import torch.nn as nn
from models.focalnet import *
from models.unet import ConvBlock
from timm.models.layers import to_3tuple


class FocalConvUNet(nn.Module):
    def __init__(self,
                 multitask=False,
                 img_size=64,
                 patch_size=2,
                 in_chans=1,
                 num_classes=1,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 focal_levels=[2, 2, 2, 2],
                 focal_windows=[3, 3, 3, 3],
                 use_conv_embed=False,
                 use_layerscale=False,
                 layerscale_value=1e-4,
                 use_postln=False,
                 use_postln_in_modulation=False,
                 normalize_modulator=False,
                 **kwargs
                 ):
        super().__init__()

        self.multitask = multitask
        self.num_layers = len(depths)
        embed_dim = [embed_dim * (2 ** i) for i in range(self.num_layers + 1)]

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.path_norm = patch_norm
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=to_3tuple(img_size),
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            use_conv_embed=use_conv_embed,
            norm_layer=norm_layer if self.path_norm else None,
            is_stem=True
        )
        self.patch_extend = PatchExtend(
            img_size=self.patch_embed.patches_resolution,
            patch_size=patch_size,
            in_chans=embed_dim[0],
            embed_dim=embed_dim[0],
            use_conv_embed=use_conv_embed,
            norm_layer=norm_layer if self.path_norm else None,
            is_stem=True
        )

        self.patches_resolution = self.patch_embed.patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder layers
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim[i_layer],
                               out_dim=embed_dim[i_layer + 1],
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer),
                                                 self.patches_resolution[2] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchEmbed,
                               focal_level=focal_levels[i_layer],
                               focal_window=focal_windows[i_layer],
                               use_conv_embed=use_conv_embed,
                               use_checkpoint=use_checkpoint,
                               use_layerscale=use_layerscale,
                               layerscale_value=layerscale_value,
                               use_postln=use_postln,
                               use_postln_in_modulation=use_postln_in_modulation,
                               normalize_modulator=normalize_modulator
                               )
            self.encoder_layers.append(layer)

        self.input_embedder = ConvBlock(in_ch=in_chans, out_ch=embed_dim[0], kernel_size=3)
        self.bottleneck_conv = ConvBlock(in_ch=embed_dim[-1], out_ch=embed_dim[-1], kernel_size=3)

        self.decoder_layers = nn.ModuleDict()
        for i in range(len(self.embed_dim) - 1):
            ind = len(self.embed_dim) - 1 - i
            self.decoder_layers[f'up-{i}'] = nn.ConvTranspose3d(in_channels=embed_dim[ind], out_channels=embed_dim[ind - 1], kernel_size=3, stride=2, padding=1, output_padding=1)
            self.decoder_layers[f'expansive-{i}'] = ConvBlock(in_ch=2 * embed_dim[ind - 1], out_ch=embed_dim[ind - 1], kernel_size=3, droprate=0.2)

        self.segmentation_head = nn.Sequential(nn.Conv3d(in_channels=2 * embed_dim[0], out_channels=num_classes, kernel_size=3, padding='same'),
                                               nn.Softmax(dim=1))

        if multitask:
            self.classification_head = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                                                     nn.Flatten(1),
                                                     nn.Linear(embed_dim[-1], num_classes),
                                                     nn.Softmax(1),
                                                     )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        shortcut_x = x
        x, D, H, W = self.patch_embed(x)
        b, L, c = x.shape
        x = self.pos_drop(x)

        residuals = [x.reshape(b, D, H, W, c).permute(0, 4, 1, 2, 3)]
        for layer in self.encoder_layers:
            x, D, H, W = layer(x, D, H, W)
            b, L, c = x.shape
            residuals.append(x.reshape(b, D, H, W, c).permute(0, 4, 1, 2, 3))

        x = residuals.pop()
        residuals.reverse()
        shortcut_bottleneck = x
        x = self.bottleneck_conv(x)
        for i in range(len(self.embed_dim) - 1):
            x = self.decoder_layers[f'up-{i}'](x)
            x = self.decoder_layers[f'expansive-{i}'](torch.cat((x, residuals[i]), dim=1))

        x, D, H, W = self.patch_extend(x)
        b, L, c = x.shape
        x = x.reshape(b, D, H, W, c).permute(0, 4, 1, 2, 3)

        if self.multitask:
            return self.classification_head(shortcut_bottleneck), self.segmentation_head(torch.cat((x, self.input_embedder(shortcut_x)), dim=1))
        else:
            return self.segmentation_head(torch.cat((x, self.input_embedder(shortcut_x)), dim=1))