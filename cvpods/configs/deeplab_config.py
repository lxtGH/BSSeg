from .base_detection_config import BaseDetectionConfig


_config_dict = dict(
    BACKBONE=dict(FREEZE_AT=0,),
    MODEL=dict(
        MASK_ON=False,
        LOAD_PROPOSALS=False,
        RESNETS=dict(
        NORM="nnSyncBN",
        OUT_FEATURES=["res5"],
        RES4_DILATION=1,
        RES5_DILATION=2,
        RES5_MULTI_GRID = [1, 2, 4],
        STEM_TYPE="deeplabv3_r50"
        ),
        SEM_SEG_HEAD=dict(
            # NAME="Deeplabv3Head",
            IGNORE_VALUE=255,
            # Number of classes in the semantic segmentation head
            NUM_CLASSES=19,
            # Number of channels in the 3x3 convs inside semantic-FPN heads.
            LOSS_TYPE="hard_pixel_mining",
            PROJECT_FEATURES=["res2"],
            PROJECT_CHANNELS=[48],
            ASPP_CHANNELS=256,
            CONVS_DIM=256,
            ASPP_DILATIONS=[6, 12, 18],
            ASPP_DROPOUT=0.1,
            USE_DEPTHWISE_SEPARABLE_CONV=False,
            COMMON_STRIDE=16,
            NORM="GN",
            LOSS_WEIGHT=1.0,
        ),
    ),

    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupPolyLR",
            WARMUP_FACTOR=1.0 / 100,
            MAX_ITER=90000,
        ),
        POLY_LR_CONSTANT_ENDING=0.0,
        POLY_LR_POWER=0.9,
    )

)


class SegmentationConfig(BaseDetectionConfig):
    def __init__(self):
        super(SegmentationConfig, self).__init__()
        self._register_configuration(_config_dict)


config = SegmentationConfig()
