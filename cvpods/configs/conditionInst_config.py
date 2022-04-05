
from .base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        # META_ARCHITECTURE="RetinaNet",
        MASK_ON=True,
        RESNETS=dict(OUT_FEATURES=["res3", "res4", "res5"]),
        FPN=dict(IN_FEATURES=["res3", "res4", "res5"]),
        FCOS=dict(
            NUM_CLASSES=80,
            IN_FEATURES=["p3", "p4", "p5", "p6", "p7"],
            FPN_STRIDES=[8, 16, 32, 64, 128],
            PRIOR_PROB=0.01,
            INFERENCE_TH_TRAIN=0.05,
            INFERENCE_TH_TEST=0.05,
            NMS_TH=0.6,
            PRE_NMS_TOPK_TRAIN=1000,
            PRE_NMS_TOPK_TEST=1000,
            POST_NMS_TOPK_TRAIN=100,
            POST_NMS_TOPK_TEST=100,
            TOP_LEVELS=2,
            NORM="GN",
            USE_SCALE=True,
            # Multiply centerness before threshold
            # This will affect the final performance by about 0.05 AP but save some time
            THRESH_WITH_CTR=True,
            # Focal loss parameters
            LOSS_ALPHA=0.25,
            LOSS_GAMMA=2.0,
            SIZES_OF_INTEREST=[64, 128, 256, 512],
            USE_RELU=True,
            USE_DEFORMABLE=False,
            # the number of convolutions used in the cls and bbox tower
            NUM_CLS_CONVS=4,
            NUM_BOX_CONVS=4,
            NUM_SHARE_CONVS=0,
            CENTER_SAMPLE=True,
            POS_RADIUS=1.5,
            LOC_LOSS_TYPE='giou',
            YIELD_PROPOSAL=False,
            YIELD_BOX_TOWER=False,
            YIELD_CLS_TOWER=False,
        ),
        CONDINST=dict(
            MASK_OUT_STRIDE=4,
            MAX_PROPOSALS=500,
            MASK_HEAD=dict(
                    CHANNELS=8,
                    NUM_LAYERS=3,
                    USE_FP16=False,
                    DISABLE_REL_COORDS=False,
                    NUM_STUFF_CLASSES=54,
                    IGNORE_VALUE=255,
            ),
            MASK_BRANCH=dict(
                    OUT_CHANNELS=8,
                    IN_FEATURES=["p3", "p4", "p5"],
                    CHANNELS=128,
                    NORM="BN",
                    NUM_CONVS=4,
                    SEMANTIC_LOSS_ON=False
            ),
            TRACKHEAD=dict(
                AMPLITUDE=0.05,
                MATCH_COEFF=[1.0, 2.0, 10.0],
                NUM_TRACK_CONVS=2,
                USE_DEFORMABLE=False,
                IN_CHANNELS=256,
                FEAT_CHANNELS=256,
                TRACK_FEAT_CHANNELS=512,
                IN_FEATURES=["p3", "p4", "p5"],
                NORM='GN',
            )
        )
    )
)


class ConditionInstConfig(BaseDetectionConfig):
    def __init__(self):
        super(ConditionInstConfig, self).__init__()
        self._register_configuration(_config_dict)


config = ConditionInstConfig()