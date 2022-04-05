from .base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        MASK_ON=True,
        PIXEL_MEAN=[103.530, 116.280, 123.675],  # BGR FORMAT
        PIXEL_STD=[1.0, 1.0, 1.0],
        RESNETS=dict(
            DEPTH=50,
            OUT_FEATURES=["res2", "res3", "res4", "res5"],
        ),
        FPN=dict(
            IN_FEATURES=["res2", "res3", "res4", "res5"],
            OUT_CHANNELS=256,
        ),
        SOLOV2=dict(
            # Instance hyper-parameters
            INSTANCE_IN_FEATURES=["p2", "p3", "p4", "p5", "p6"],
            FPN_INSTANCE_STRIDES=[8, 8, 16, 32, 32],
            FPN_SCALE_RANGES=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
            SIGMA=0.2,
            # Channel size for the instance head.
            INSTANCE_IN_CHANNELS=256,
            INSTANCE_CHANNELS=512,
            # Convolutions to use in the instance head.
            NUM_INSTANCE_CONVS=4,
            USE_DCN_IN_INSTANCE=False,
            TYPE_DCN='DCN',
            NUM_GRIDS=[40, 36, 24, 16, 12],
            # Number of foreground classes.
            NUM_CLASSES=80,
            NUM_KERNELS=256,
            NORM="GN",
            USE_COORD_CONV=True,
            PRIOR_PROB=0.01,
            # Mask hyper-parameters.
            # Channel size for the mask tower.
            MASK_IN_FEATURES=["p2", "p3", "p4", "p5"],
            MASK_IN_CHANNELS=256,
            MASK_CHANNELS=128,
            NUM_MASKS=256,     # NUM_MASKS * kernel_size**2 = NUM_KERNELS
            # Test cfg.
            NMS_PRE=500,
            SCORE_THR=0.1,
            UPDATE_THR=0.05,
            MASK_THR=0.5,
            MAX_PER_IMG=100,
            # NMS type: matrix OR mask.
            NMS_TYPE="matrix",
            NMS_KERNEL="gaussian",
            NMS_SIGMA=2,
            # Loss cfg.
            LOSS=dict(
                FOCAL_USE_SIGMOID=True,
                FOCAL_ALPHA=0.25,
                FOCAL_GAMMA=2.0,
                FOCAL_WEIGHT=1.0,
                DICE_WEIGHT=3.0
            )
        ),
    ),
    INPUT=dict(
        # SOLO for instance segmenation does not work with "polygon" mask_format
        MASK_FORMAT="bitmask",
    )
)


class SOLOV2Config(BaseDetectionConfig):
    def __init__(self):
        super(SOLOV2Config, self).__init__()
        self._register_configuration(_config_dict)


config = SOLOV2Config()
