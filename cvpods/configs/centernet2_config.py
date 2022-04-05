#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   base-centernet2.py
@Author             :   Xiangtai Li
'''


from .rcnn_config import RCNNConfig

_config_dict = dict(
    DEBUG=False,
    SAVE_DEBUG=False,
    SAVE_PTH=False,
    VIS_THRESH=0.3,
    DEBUG_SHOW_NAME=False,

    MODEL=dict(
        RESNETS=dict(OUT_FEATURES=["res3", "res4", "res5"],),
        FPN=dict(IN_FEATURES=["res3", "res4", "res5"]),
        PROPOSAL_GENERATOR=dict(
            # Current proposal generators include "RPN", "RRPN" and "PrecomputedProposals"
            NAME="CenterNet",
            MIN_SIZE=0,
        ),
        ROI_HEADS=dict(
            NAME="CustomCascadeROIHeads",
            IN_FEATURES=["p3", "p4", "p5", "p6", "p7"],
            IOU_THRESHOLDS=[0.6],
            NMS_THRESH_TEST=0.7
        ),

        ROI_BOX_CASCADE_HEAD=dict(
            # The number of cascade stages is implicitly defined by
            # the length of the following two configs.
            BBOX_REG_WEIGHTS=(
                (10.0, 10.0, 5.0, 5.0),
                (20.0, 20.0, 10.0, 10.0),
                (30.0, 30.0, 15.0, 15.0),
            ),
            IOUS=(0.6, 0.7, 0.8),
        ),

        ROI_BOX_HEAD=dict(
            NUM_FC=2,
            POOLER_RESOLUTION=7,
            CLS_AGNOSTIC_BBOX_REG=True,
            USE_SIGMOID_CE=False,
            PRIOR_PROB=0.01,
            USE_EQL_LOSS=False,
            CAT_FREQ_PATH='datasets/lvis/lvis_v1_train_cat_info.json',
            EQL_FREQ_CAT=200,
            USE_FED_LOSS=False,
            FED_LOSS_NUM_CAT=50,
            FED_LOSS_FREQ_WEIGHT=0.5,
            MULT_PROPOSAL_SCORE=False,
        ),
        CENTERNET=dict(
            NUM_CLASSES=80,
            IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"],
            FPN_STRIDES = [8, 16, 32, 64, 128],
            PRIOR_PROB = 0.01,
            INFERENCE_TH = 0.05,
            CENTER_NMS = False,
            NMS_TH_TRAIN = 0.6,
            NFERENCE_TH = 0.05,
            NMS_TH_TEST = 0.6,
            PRE_NMS_TOPK_TRAIN = 1000,
            POST_NMS_TOPK_TRAIN = 100,
            PRE_NMS_TOPK_TEST = 1000,
            POST_NMS_TOPK_TEST = 100,
            NORM = "GN",
            USE_DEFORMABLE = False,
            NUM_CLS_CONVS = 4,
            NUM_BOX_CONVS = 4,
            NUM_SHARE_CONVS = 0,
            LOC_LOSS_TYPE = 'giou',
            SIGMOID_CLAMP = 1e-4,
            HM_MIN_OVERLAP = 0.8,
            MIN_RADIUS = 4,
            SOI = [[0, 80], [64, 160], [128, 320], [256, 640], [512, 10000000]],
            POS_WEIGHT = 1.,
            NEG_WEIGHT = 1.,
            REG_WEIGHT = 2.,
            HM_FOCAL_BETA = 4,
            HM_FOCAL_ALPHA = 0.25,
            LOSS_GAMMA = 2.0,
            WITH_AGN_HM = False,
            ONLY_PROPOSAL = False,
            AS_PROPOSAL = False,
            IGNORE_HIGH_FP = -1.,
            MORE_POS = False,
            MORE_POS_THRESH = 0.2,
            MORE_POS_TOPK = 9,
            NOT_NORM_REG = True,
            NOT_NMS = False,
        ),

        ROI_MASK_HEAD=dict(
            # NAME: "MaskRCNNConvUpsampleHead"
            NUM_CONV=4,
            POOLER_RESOLUTION=14,
        ),

        BIFPN=dict(
            NUM_LEVELS=5,
            NUM_BIFPN=6,
            NORM='GN',
            OUT_CHANNELS=160,
            SEPARABLE_CONV=False,
        ),



        ROI_TRACK_HEAD=dict(
            # NAME: "TrackHead"
            POOLER_RESOLUTION=7,
            PID_WEIGHT=-1,
        ),
    ),
)


class CenterNet2Config(RCNNConfig):
    def __init__(self):
        super(CenterNet2Config, self).__init__()
        self._register_configuration(_config_dict)


config = CenterNet2Config()
