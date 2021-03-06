#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   retinanet_config.py
@Time               :   2020/05/07 23:56:02
@Author             :   Benjin Zhu
@Contact            :   poodarchu@gmail.com
@Last Modified by   :   Benjin Zhu
@Last Modified time :   2020/05/07 23:56:02
'''

from .base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        # Backbone NAME: "build_retinanet_resnet_fpn_backbone"
        RESNETS=dict(OUT_FEATURES=["res3", "res4", "res5"]),
        FPN=dict(IN_FEATURES=["res3", "res4", "res5"]),
        ANCHOR_GENERATOR=dict(
            SIZES=[
                [x, x * 2 ** (1.0 / 3), x * 2 ** (2.0 / 3)]
                for x in [32, 64, 128, 256, 512]
            ]
        ),
        RETINANET=dict(
            # This is the number of foreground classes.
            NUM_CLASSES=80,
            IN_FEATURES=["p3", "p4", "p5", "p6", "p7"],
            # Convolutions to use in the cls and bbox tower
            # NOTE: this doesn't include the last conv for logits
            NUM_CONVS=0,
            # IoU overlap ratio [bg, fg] for labeling anchors.
            # Anchors with < bg are labeled negative (0)
            # Anchors  with >= bg and < fg are ignored (-1)
            # Anchors with >= fg are labeled positive (1)
            IOU_THRESHOLDS=[0.4, 0.5],
            IOU_LABELS=[0, -1, 1],
            # Prior prob for rare case (i.e. foreground) at the beginning of training.
            # This is used to set the bias for the logits layer of the classifier subnet.
            # This improves training stability in the case of heavy class imbalance.
            PRIOR_PROB=0.01,
            # Inference cls score threshold, only anchors with score > INFERENCE_TH are
            # considered for inference (to improve speed)
            SCORE_THRESH_TEST=0.05,
            TOPK_CANDIDATES_TEST=1000,
            NMS_THRESH_TEST=0.5,
            # Weights on (dx, dy, dw, dh) for normalizing Retinanet anchor regression targets
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            # Loss parameters
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            SMOOTH_L1_LOSS_BETA=0.11,
        ),
        SEPC=dict(
            IN_FEATURES=["p3", "p4", "p5", "p6", "p7"],
            IN_CHANNELS=[256, 256, 256, 256, 256],
            OUT_CHANNELS=256,
            NUM_OUTS=5,
            COMBINE_DEFORM=False,
            EXTRA_DEFORM=False,
            COMBINE_NUM=4,
            IBN=False,
        )
    ),
)


class RetinaNetSEPCConfig(BaseDetectionConfig):
    def __init__(self):
        super(RetinaNetSEPCConfig, self).__init__()
        self._register_configuration(_config_dict)


config = RetinaNetSEPCConfig()
