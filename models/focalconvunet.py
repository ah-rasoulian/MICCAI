import torch
import torch.nn as nn
from models.focalnet import *
from models.unet import ResConvBlock
from timm.models.layers import to_3tuple
import torch.nn.functional as F


class FocalConvUNet(nn.Module):
    def __init__(self,
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

        self.patches_resolution = self.patch_embed.patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder layers
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.encoder_layers.append(BasicLayer(dim=embed_dim[i_layer],
                                                  out_dim=embed_dim[i_layer + 1],
                                                  input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                                    self.patches_resolution[1] // (2 ** i_layer),
                                                                    self.patches_resolution[2] // (2 ** i_layer)),
                                                  depth=depths[i_layer],
                                                  mlp_ratio=self.mlp_ratio,
                                                  drop=drop_rate if self.num_layers - 1 - i_layer <= 1 else 0.,
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
                                                  ))

        self.residual_conv_layers = nn.ModuleList()
        for i_layer in range(self.num_layers + 1):
            in_ch = in_chans if i_layer == 0 else embed_dim[i_layer - 1]
            out_ch = embed_dim[0] if i_layer == 0 else in_ch
            self.residual_conv_layers.append(ResConvBlock(in_ch, out_ch, 3, 0))

        self.bottleneck_conv = ResConvBlock(in_ch=embed_dim[-1], out_ch=embed_dim[-1], kernel_size=3)

        self.decoder_layers = nn.ModuleList()
        for i in range(len(self.embed_dim)):
            ind = len(self.embed_dim) - 1 - i
            in_ch = embed_dim[ind]
            out_ch = embed_dim[ind - 1] if ind > 0 else embed_dim[0]
            self.decoder_layers.append(FocalConvUpBlock(in_ch, out_ch, drop_rate=drop_rate if i <= 1 else 0.))

        self.segmentation_head = nn.Conv3d(in_channels=embed_dim[0], out_channels=num_classes, kernel_size=3, padding='same')

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

        residuals = [shortcut_x, reshape_tokens_to_volumes(x, D, H, W, True)]
        for layer in self.encoder_layers:
            x, D, H, W = layer(x, D, H, W)
            residuals.append(reshape_tokens_to_volumes(x, D, H, W, True))

        x = residuals.pop()
        residuals.reverse()
        x = self.bottleneck_conv(x)

        for i in range(len(self.embed_dim)):
            x = self.decoder_layers[i](x, self.residual_conv_layers[len(self.embed_dim) - 1 - i](residuals[i]))

        return self.segmentation_head(x)


class FocalConvUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, drop_rate=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.expansive = ResConvBlock(2 * out_ch, out_ch, kernel_size, drop_rate)

    def forward(self, x, skip):
        x = self.up(x)
        concat = torch.cat((x, skip), dim=1)
        x = self.expansive(concat)
        return x


def reshape_tokens_to_volumes(x: torch.Tensor, d, h, w, normalize=False):
    b, L, c = x.shape
    x = x.reshape(b, d, h, w, c)
    if normalize:
        x = F.layer_norm(x, [c])
    x = x.permute(0, 4, 1, 2, 3)
    return x
