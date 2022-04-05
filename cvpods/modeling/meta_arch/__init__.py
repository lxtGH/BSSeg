# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# import all the meta_arch, so they will be registered

from .centernet import CenterNet
from .borderdet import BorderDet
from .panoptic_fpn import PanopticFPN
from .rcnn import GeneralizedRCNN, ProposalNetwork
from .reppoints import RepPoints
from .semantic_seg import SemanticSegmentor, SemSegFPNHead
from .ssd import SSD
from .tensormask import TensorMask
from .yolov3 import YOLOv3

from .solo.solo import SOLO
from .solo.solov2 import SOLOv2
from .solo.solo_decoupled import DecoupledSOLO
from cvpods.modeling.meta_arch.conditionalInst.conditionalInst import CondInst
from cvpods.modeling.meta_arch.sparsercnn.sparse_rcnn import SparseRCNN
from cvpods.modeling.meta_arch.retinanet.retinanet_sepc import RetinaNetSEPC
from cvpods.modeling.meta_arch.retinanet.retinanet import RetinaNet
from cvpods.modeling.meta_arch.fcos.fcos import FCOS
from cvpods.modeling.meta_arch.fcos.fcos_sepc import FCOSSEPC
from cvpods.modeling.meta_arch.detr.detr import DETR


from .efficientdet import EfficientDet
from .pointrend import (
    PointRendROIHeads,
    CoarseMaskHead,
    StandardPointHead,
    PointRendSemSegHead,
)
from .dynamic4seg import DynamicNet4Seg
from .fcn import FCNHead
