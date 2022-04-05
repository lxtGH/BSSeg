#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from .rcnn_config import RCNNConfig

_config_dict = dict(
    MODEL=dict(
        IGNORE_VALUE=255,
        TENSOR_DIM=100,
        RESNETS=dict(OUT_FEATURES=["res2", "res3", "res4", "res5"],),
        FPN=dict(IN_FEATURES=["res2", "res3", "res4", "res5"]),

        POSITION_HEAD=dict(
            NUM_CONVS=3,
            COORD=False,
            CONVS_DIM=256,
            NORM="GN",
            DEFORM=True,
            THING=dict(
                CENTER_TYPE="mass",
                POS_NUM=7,
                NUM_CLASSES=80,
                BIAS_VALUE=-2.19,
                MIN_OVERLAP=0.7,
                GAUSSIAN_SIGMA=3,
                THRES=0.05,
                TOP_NUM=100
            ),
            STUFF=dict(
                NUM_CLASSES=54,
                WITH_THING=True,
                THRES=0.05
            )

        ),
        SEM_SEG_HEAD=dict(
            # i.e., no loss is calculated for the correposnding pixel.
            IGNORE_VALUE=255,
            # Number of classes in the semantic segmentation head
            NUM_CLASSES=54,
        ),

        KERNEL_HEAD=dict(
            INSTANCE_SCALES=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048),),
            TEST_SCALES=((1, 64), (64, 128), (128, 256), (256, 512), (512, 2048),),
            NUM_CONVS=3,
            DEFORM=False,
            COORD=True,
            CONVS_DIM=256,
            NORM="GN"
        ),

        FEATURE_ENCODER=dict(
            IN_FEATURES=["p3", "p4", "p5", "p6", "p7"],
            NUM_CONVS=3,
            CONVS_DIM=64,
            DEFORM=False,
            COORD=True,
            NORM=""
        ),

        SEMANTIC_FPN=dict(
            IN_FEATURES=["p2", "p3", "p4", "p5"],
            CONVS_DIM=256,
            COMMON_STRIDE=4,
            NORM="GN"
        ),

        LOSS_WEIGHT=dict(
            POSITION=1.0,
            SEGMENT=3.0,
            FOCAL_LOSS_ALPHA=0.25,
            FOCAL_LOSS_GAMMA=2.0
        ),

        INFERENCE=dict(
            INST_THRES=0.4,
            SIMILAR_THRES=0.9,
            SIMILAR_TYPE="cosine",
            CLASS_SPECIFIC=True,
            COMBINE=dict(
                ENABLE=True,
                NO_OVERLAP=False,
                OVERLAP_THRESH=0.5,
                STUFF_AREA_LIMIT=4096,
                #INSTANCES_CONFIDENCE_THRESH=0.5,
                INST_THRESH=0.2
            ),
        ),
    ),
    SOLVER=dict(
        OPTIMIZER=dict(
            NAME="SGD",
            BASE_LR=0.01,
        ),
        LR_SCHEDULER=dict(
            NAME="WarmupPolyLR",
            POLY_LR_POWER=0.9,
            WARMUP_FACTOR=1.0 / 1000,
            WARMUP_ITERS=1000,
            MAX_ITER=90000,
        ),
        CLIP_GRADIENTS=dict(
            ENABLED=True,
            # - "value": the absolute values of elements of each gradients are clipped
            CLIP_TYPE="value",
            # Maximum absolute value used for clipping gradients
            CLIP_VALUE=35.0,
        ),
        POLY_LR_CONSTANT_ENDING=0.0,
        POLY_LR_POWER=0.9,
    ),
    INPUT=dict(
        MASK_FORMAT="bitmask"
    )
)


class PANFCNConfig(RCNNConfig):
    def __init__(self):
        super(PANFCNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = PANFCNConfig()
