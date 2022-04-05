import os.path as osp

from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="/mnt/lustre/share_data/chengguangliang/hehao/CEInst_checkpoint/r50_3x.pth",
        MASK_ON=True,
        RESNETS=dict(DEPTH=50),
    ROI_HEADS=dict(NUM_CLASSES=8),
    BOUNDARY_MASK_HEAD=dict(
        OUTPUT_RATIO=1,
        POOLER_RESOLUTION=28,
        IN_FEATURES=["p2"],
        NUM_CONV=2),
        ROI_MASK_HEAD=dict(
            CEMODULE=dict(
                NUM_CONV=2,
                PLANES=256,
                DCN_ON=True,
                DCN_V2=True,
                NUM_EDGE_CONV=2,
                FUSE_MODE="Add",
                WITH_EDGE_REFINE=True,
                NORM='GN',
                KERNEL_SIZE=5
            ),
            LOSS_WEIGHT=[1.0, 1.0, 1.0, 1.0]
        ),
    ),
    DATASETS=dict(
        TRAIN=("cityscapes_fine_instance_seg_train",),
        TEST=("cityscapes_fine_instance_seg_val",),
    ),
    SOLVER=dict(
        IMS_PER_BATCH=8,
        IMS_PER_DEVICE=1,
        LR_SCHEDULER=dict(
            STEPS=(18000,),
            MAX_ITER=24000,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.01,
        ),
        CHECKPOINT_PERIOD=8000,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=(800, 832, 864, 896, 928, 960, 992, 1024),
                    max_size=2048, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=1024, max_size=2048, sample_style="choice")),
            ],
        ),
    ),
    TEST=dict(
        EVAL_PEROID=10000,
    ),
    OUTPUT_DIR="output"
)


class FasterRCNNConfig(RCNNFPNConfig):
    def __init__(self):
        super(FasterRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = FasterRCNNConfig()
