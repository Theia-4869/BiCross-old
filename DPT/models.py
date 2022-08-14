import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import StdConv2dSame

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)
from .functions import ReverseLayerF

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT(BaseModel):
    def __init__(
        self,
        head=None,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
        spike_input=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        if spike_input:
            self.pretrained.model.patch_embed.backbone.stem.conv = StdConv2dSame(128, 64, kernel_size=(7, 7), stride=(2, 2), bias=False)
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4, cls_token = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out, path_1, path_2, path_3, path_4, layer_4, cls_token


class DPTDepthModel(DPT):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, with_uncertainty=False, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert
        self.with_uncertainty = with_uncertainty

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head=head, **kwargs)

        if with_uncertainty:
            self.uncertaity_head = nn.Sequential(
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                # nn.ReLU(True),
                # nn.Sigmoid(),
                # nn.Softplus(),
            )

            # for m in self.uncertaity_head.modules():
            #     if isinstance(m, torch.nn.Conv2d):
            #         torch.nn.init.xavier_uniform_(m.weight, gain = torch.nn.init.calculate_gain('relu'))
            #         torch.nn.init.kaiming_normal_(m.weight, mode = 'fan_in', nonlinearity='relu')

        if path is not None:
            print("loading checkpoint from", path, "...")
            self.load(path)

    def forward(self, x):
        inv_depth, path_1, path_2, path_3, path_4, layer_4, cls_token = super().forward(x)
        inv_depth = inv_depth.squeeze(dim=1)

        if self.with_uncertainty:
            uncertainty = self.uncertaity_head(path_1).squeeze(dim=1)
            if self.invert:
                depth = self.scale * inv_depth + self.shift
                depth[depth < 1e-8] = 1e-8
                depth = 1.0 / depth
                return depth, uncertainty, path_1, path_2, path_3, path_4, layer_4, cls_token
            else:
                return inv_depth, uncertainty, path_1, path_2, path_3, path_4, layer_4, cls_token

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            return depth, path_1, path_2, path_3, path_4, layer_4, cls_token
        else:
            return inv_depth, path_1, path_2, path_3, path_4, layer_4, cls_token
