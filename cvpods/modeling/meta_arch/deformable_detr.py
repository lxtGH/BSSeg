# To Do: implement the deformable detr
# Xiangtai Li
import torch
import torch.nn.functional as F
from torch import nn

from cvpods.layers import ShapeSpec, position_encoding_dict
from cvpods.modeling.backbone import Transformer
from cvpods.modeling.matcher import HungarianMatcher
from cvpods.structures import Boxes, ImageList, Instances
from cvpods.structures import boxes as box_ops
from cvpods.layers.box_ops import generalized_box_iou
from cvpods.utils import comm
from cvpods.layers.misc import accuracy


