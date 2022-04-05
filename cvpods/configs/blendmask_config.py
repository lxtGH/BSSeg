
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
        BLENDMASK=dict(
            MASK_OUT_STRIDE=4,
            ATTN_SIZE=14,
            TOP_INTERP="bilinear",
            BOTTOM_RESOLUTION=56,
            POOLER_TYPE="ROIAlignV2",
            POOLER_SAMPLING_RATIO=1,
            POOLER_SCALES=(0.25,),
            INSTANCE_LOSS_WEIGHT=1.0,
            VISUALIZE=False,
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
            ),
        ),
        BASIS_MODULE=dict(
            NAME="ProtoNet",
            NUM_BASES=4,
            LOSS_ON=False,
            CONDTION_INST_LOSS_ON=False,
            ANN_SET="coco",
            CONVS_DIM=128,
            IN_FEATURES=["p3", "p4", "p5"],
            NORM="SyncBN",
            NUM_CONVS=3,
            COMMON_STRIDE=8,
            NUM_CLASSES=80,
            LOSS_WEIGHT=0.3
        ),

        PANOPTIC_FPN=dict(
            # Scaling of all losses from instance detection / segmentation head.
            INSTANCE_LOSS_WEIGHT=1.0,
            # options when combining instance & semantic segmentation outputs
            COMBINE=dict(
                ENABLED=False,  # by default we do not use the panoptic segmentation
                OVERLAP_THRESH=0.5,
                STUFF_AREA_LIMIT=4096,
                INSTANCES_CONFIDENCE_THRESH=0.5,
            ),
        ),
    )
)


class BlendMaskConfig(BaseDetectionConfig):
    def __init__(self):
        super(BlendMaskConfig, self).__init__()
        self._register_configuration(_config_dict)


config = BlendMaskConfig()