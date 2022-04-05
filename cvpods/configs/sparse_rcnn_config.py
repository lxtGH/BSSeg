#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Xiangtai Li
'''

from .rcnn_config import RCNNConfig

_config_dict = dict(
    MODEL=dict(
        # BACKBONE=dict(NAME='build_resnet_backbone',),
        RESNETS=dict(OUT_FEATURES=["res2", "res3", "res4", "res5"],),
        FPN=dict(IN_FEATURES=["res2", "res3", "res4", "res5"]),
        ROI_HEADS=dict(
            # NAME: "StandardROIHeads"
            IN_FEATURES=["p2", "p3", "p4", "p5"],
        ),
        ROI_BOX_HEAD=dict(
            POOLER_TYPE="ROIAlignV2",
            POOLER_SAMPLING_RATIO=2,
            POOLER_RESOLUTION=7,
        ),
        SparseRCNN=dict(
            NUM_PROPOSALS=100,
            NUM_CLASSES=80,
            NHEADS=8,
            DROPOUT=0.0,
            DIM_FEEDFORWARD=2048,
            ACTIVATION='relu',
            HIDDEN_DIM=256,
            NUM_CLS=1,
            NUM_REG=3,
            NUM_HEADS=6,

            # Dynamic Conv.
            NUM_DYNAMIC=2,
            DIM_DYNAMIC=64,

            # Loss.
            CLASS_WEIGHT=2.0,
            GIOU_WEIGHT=2.0,
            L1_WEIGHT=5.0,
            DEEP_SUPERVISION=True,
            NO_OBJECT_WEIGHT=0.1,
            USE_FOCAL=True,
            ALPHA=0.25,
            GAMMA=2.0,
            PRIOR_PROB=0.01
        )
    ),
)


class SparseRCNNFPNConfig(RCNNConfig):
    def __init__(self):
        super(SparseRCNNFPNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = SparseRCNNFPNConfig()
