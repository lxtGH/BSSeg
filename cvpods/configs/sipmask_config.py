
from .base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
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
        SIPMASK=dict(
            INFERENCE_TH=0.05,
            PRE_NMS_TOP_N=1000,
            NMS_TH=0.6,
            POST_NMS_TOP_N=100,
            INFERENCE_TH_TRACK=0.03,
            PRE_NMS_TOP_N_TRACK=200,
            NMS_TH_TRACK=0.5,
            POST_NMS_TOP_N_TRACK=10,
            THRESH_WITH_CTR=True,
            FPN_STRIDES=[8, 16, 32, 64, 128],
            TRACK_ON=False,
            HEAD=dict(
                NUM_BASIC_MASKS=32,
                SUB_REGION_X=2,
                SUB_REGION_Y=2,
                IN_CHANNELS=256,
                DEFORMABLE_GROUPS=4,
                FA_KERNEL_SIZE=3,
                MASK_THRESH=0.4,
                CONDTION_INST_LOSS_ON=False,
            ),
            TRACKHEAD=dict(
                AMPLITUDE=0.05,
                NUM_TRACK_CONVS=2,
                USE_DEFORMABLE=False,
                IN_CHANNELS=256,
                FEAT_CHANNELS=256,
                NORM='GN',
                MATCH_COEFF=[1.0, 2.0, 10.0],
                IN_FEATURES=["p3", "p4", "p5"],
            )
        )
    )
)


class SipMaskConfig(BaseDetectionConfig):
    def __init__(self):
        super(SipMaskConfig, self).__init__()
        self._register_configuration(_config_dict)


config = SipMaskConfig()