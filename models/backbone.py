# ------------------------------------------------------------------------
# Modified from UP-DETR  https://github.com/dddzg/up-detr
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

#from util.misc import NestedTensor
#from .position_encoding import build_position_encoding
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_layer: str):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            #if not train_backbone:
                parameter.requires_grad_(False)
        
        return_layers = {return_layer: '0'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list):
        """supports both NestedTensor and torch.Tensor
        """
        out = self.body(tensor_list)
        return out['0']

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_layer: str,
                 frozen_bn: bool,
                 dilation: bool):
        
        if frozen_bn:
            backbone = getattr(torchvision.models, name)(
                               replace_stride_with_dilation=[False, False, dilation],
                               pretrained=True, norm_layer=FrozenBatchNorm2d)
        else:
            backbone = getattr(torchvision.models, name)(
                               replace_stride_with_dilation=[False, False, dilation],
                               pretrained=True)
            
        # load the SwAV pre-training model from the url instead of supervised pre-training model
        if name == 'resnet50':
            checkpoint = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            backbone.load_state_dict(state_dict, strict=False)
            #pass
        if name in ('resnet18', 'resnet34'):
            num_channels = 512
        else:
            if return_layer == 'layer3':
                num_channels = 1024
            else:
                num_channels = 2048
        super().__init__(backbone, train_backbone, num_channels, return_layer)

def build_backbone(cfg):
    train_backbone = cfg.TRAIN.lr_backbone > 0
    backbone = Backbone(cfg.MODEL.backbone, train_backbone, cfg.MODEL.backbone_layer, cfg.MODEL.fix_bn, cfg.MODEL.dilation)
    return backbone

if __name__ == '__main__':
    backbone = Backbone('resnet50',
                        train_backbone=True,
                        return_layer='layer3',
                        frozen_bn=False,
                        dilation=False)
    
    inputs = torch.rand(5,3,256,256)
    outputs = backbone(inputs)
