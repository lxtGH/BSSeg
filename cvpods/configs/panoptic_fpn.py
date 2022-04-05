#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from .rcnn_config import RCNNConfig

_config_dict = dict(
    MODEL=dict(
        RESNETS=dict(OUT_FEATURES=["res2", "res3", "res4", "res5"],),
        FPN=dict(IN_FEATURES=["res2", "res3", "res4", "res5"]),
        ANCHOR_GENERATOR=dict(
            SIZES=[[32], [64], [128], [256], [512]], ASPECT_RATIOS=[[0.5, 1.0, 2.0]],
        ),
        RPN=dict(
            IN_FEATURES=["p2", "p3", "p4", "p5", "p6"],
            PRE_NMS_TOPK_TRAIN=2000,
            PRE_NMS_TOPK_TEST=1000,
            POST_NMS_TOPK_TRAIN=1000,
            POST_NMS_TOPK_TEST=1000,
        ),
        ROI_HEADS=dict(
            # NAME: "StandardROIHeads"
            IN_FEATURES=["p2", "p3", "p4", "p5"],
        ),
        ROI_BOX_HEAD=dict(
            # NAME: "FastRCNNConvFCHead"
            NUM_FC=2,
            POOLER_RESOLUTION=7,
        ),
        ROI_MASK_HEAD=dict(
            # NAME: "MaskRCNNConvUpsampleHead"
            NUM_CONV=4,
            POOLER_RESOLUTION=14,
        ),
        SEM_SEG_HEAD=dict(
            # NAME="SemSegFPNHead",
            IN_FEATURES=["p2", "p3", "p4", "p5"],
            # Label in the semantic segmentation ground truth that is ignored,
            # i.e., no loss is calculated for the correposnding pixel.
            IGNORE_VALUE=255,
            # Number of classes in the semantic segmentation head
            NUM_CLASSES=54,
            # Number of channels in the 3x3 convs inside semantic-FPN heads.
            CONVS_DIM=128,
            # Outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
            COMMON_STRIDE=4,
            # Normalization method for the convolution layers. Options: "" (no norm), "GN".
            NORM="GN",
            LOSS_WEIGHT=1.0,
        ),
        PANOPTIC_FPN=dict(
            # Scaling of all losses from instance detection / segmentation head.
            INSTANCE_LOSS_WEIGHT=1.0,
            # options when combining instance & semantic segmentation outputs
            COMBINE=dict(
                ENABLED=True,
                OVERLAP_THRESH=0.5,
                STUFF_AREA_LIMIT=4096,
                INSTANCES_CONFIDENCE_THRESH=0.5,
            ),
        ),

    ),
)


class PANFPNConfig(RCNNConfig):
    def __init__(self):
        super(PANFPNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = PANFPNConfig()
