from .base_detection_config import BaseDetectionConfig


_config_dict = dict(
    BACKBONE=dict(FREEZE_AT=-1,),
    DATALOADER=dict(
            # Number of data loading threads
            NUM_WORKERS=10,
            # Default sampler for dataloader
            SAMPLER_TRAIN="DistributedGroupSampler",
            # Repeat threshold for RepeatFactorTrainingSampler
            REPEAT_THRESHOLD=0.0,
            # If True, the dataloader will filter out images that have no associated
            # annotations at train time.
            FILTER_EMPTY_ANNOTATIONS=True,
        ),
    MODEL=dict(
        MASK_ON=False,
        LOAD_PROPOSALS=False,
        RESNETS=dict(
            NORM="SyncBN",
            OUT_FEATURES=["res2", "res3", "res5"],
            RES4_DILATION=1,
            RES5_DILATION=2,
            STRIDE_IN_1X1=False,
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
            IN_FEATURES=["res2", "res3", "res5"],
            PROJECT_FEATURES=["res2", "res3"],
            PROJECT_CHANNELS=[32, 64],
            ASPP_CHANNELS=256,
            CONVS_DIM=256,
            ASPP_DILATIONS=[6, 12, 18],
            ASPP_DROPOUT=0.1,
            USE_DEPTHWISE_SEPARABLE_CONV=False,
            COMMON_STRIDE=4,
            NORM="nnSyncBN",
            LOSS_WEIGHT=1.0,
            HEAD_CHANNELS=256,
            LOSS_TOP_K=0.2
        ),
        INS_EMBED_HEAD=dict(
            IN_FEATURES=["res2", "res3", "res5"],
            PROJECT_FEATURES=["res2", "res3"],
            PROJECT_CHANNELS=[32, 64],
            ASPP_CHANNELS=256,
            ASPP_DILATIONS=[6, 12, 18],
            ASPP_DROPOUT=0.1,
            # We add an extra convolution before predictor.
            HEAD_CHANNELS=32,
            CONVS_DIM=128,
            COMMON_STRIDE=4,
            NORM="SyncBN",
            CENTER_LOSS_WEIGHT=200.0,
            OFFSET_LOSS_WEIGHT=0.01,
        ),
        PANOPTIC_DEEPLAB=dict(
            STUFF_AREA=2048,
            CENTER_THRESHOLD=0.1,
            NMS_KERNEL=7,
            TOP_K_INSTANCE=200,
            # If set to False, Panoptic-DeepLab will not evaluate instance segmentation.
            PREDICT_INSTANCES=True,
            USE_DEPTHWISE_SEPARABLE_CONV=False,
            # This is the padding parameter for images with various sizes. ASPP layers
            # requires input images to be divisible by the average pooling size and we
            # can use `MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY` to pad all images to
            # a fixed resolution (e.g. 640x640 for COCO) to avoid having a image size
            # that is not divisible by ASPP average pooling size.
            SIZE_DIVISIBILITY=-1,
            # Only evaluates network speed (ignores post-processing).
            BENCHMARK_NETWORK_SPEED=False,
        )

    ),
    SOLVER=dict(
        OPTIMIZER=dict(
            NAME="Adam",
            BASE_LR=0.001,
            BASE_LR_RATIO_BACKBONE=0.1,
            WEIGHT_DECAY=1e-4,
        ),
        LR_SCHEDULER=dict(
            NAME="WarmupPolyLR",
            WARMUP_FACTOR=1.0 / 1000,
            MAX_ITER=90000,
            STEPS=(30000, )
        ),
        POLY_LR_CONSTANT_ENDING=0.0,
        POLY_LR_POWER=0.9,
    ),
    INPUT=dict(
        FORMART="RGB",
        GAUSSIAN_SIGMA=10,
        IGNORE_STUFF_IN_OFFSET=True,
        SMALL_INSTANCE_AREA=4096,
        SMALL_INSTANCE_WEIGHT=3,
        IGNORE_CROWD_IN_SEMANTIC=False,
    )
)


class PanoptocDeeplab(BaseDetectionConfig):
    def __init__(self):
        super(PanoptocDeeplab, self).__init__()
        self._register_configuration(_config_dict)


config = PanoptocDeeplab()