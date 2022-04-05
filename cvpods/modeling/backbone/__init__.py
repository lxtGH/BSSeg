# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .backbone import Backbone
from .fpn import FPN, build_retinanet_resnet_fpn_p5_backbone
from .resnet import ResNet, ResNetBlockBase, build_resnet_backbone, make_stage, build_resnet_deeplab_backbone
from .darknet import Darknet, build_darknet_backbone
from .efficientnet import EfficientNet, build_efficientnet_backbone
from .bifpn import BiFPN, build_efficientnet_bifpn_backbone
from .dynamic_arch import DynamicNetwork, build_dynamic_backbone
from .sf_fpn import build_resnet_sf_fpn_backbone
from .transformer import Transformer
from .swin import build_swin_backbone, build_swin_fpn_backbone, build_retinanet_swin_fpn_backbone
# TODO can expose more resnet blocks after careful consideration
