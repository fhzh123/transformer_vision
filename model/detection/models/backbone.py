import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from model.detection.util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.

    The reason why we use FrozenBatchNorm2d instead of BatchNorm2d is that the 
    sizes of the batches are very small, which makes the batch statistics very 
    poor and degrades performance. Plus, when using multiple GPUs, the batch 
    statistics are not accumulated from multiple devices, so that only a single 
    GPU compute the statistics.
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

    def __init__(self, 
                backbone: nn.Module, 
                train_backbone: bool, 
                num_channels: int, 
                return_interm_layers: bool):
        super().__init__()

        # freeze할지 말지 여부 
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        # return_layer에 따라 가져올 중간 layer의 이름과 새로운 이름 
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):


        # ex) xs : {layer name : feature map}
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None

            # mask를 feature map의 width, height 크기로 interpolate 
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        # ex) out : {layer name : NestedTensor(feature map, feature map sized mask)}
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, 
                 name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):

        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)

        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):

        # self[0] : backbone
        # self[1] : position_embedding

        xs = self[0](tensor_list) # xs : out : {layer name : NestedTensor(feature map, feature map sized mask)}
        out: List[NestedTensor] = []
        pos = []

        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        # for o, p in zip(out, pos):
        #     print("Backbone output shape :", o.tensors.shape)
        #     print("Position Embeding layer output shape :", p.shape)
        """
        Returns :

        batch_size = 2
        image shape : torch.Size([3, 672, 778])
                      torch.Size([3, 704, 997])
        output(feature map) shape : torch.Size([2, 2048, 24, 37]) 
        position embedding shape : torch.Size([2, 256, 24, 37])
        """

        return out, pos


def build_backbone(args):

    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model