import os.path as osp

from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="/mnt/lustreold/share_data/lixiangtai/pretrained/R-50.pkl",
        MASK_ON=True,
        RESNETS=dict(DEPTH=50),
    BOUNDARY_MASK_HEAD=dict(
        OUTPUT_RATIO=1,
        POOLER_RESOLUTION=28,
        IN_FEATURES=["p2"],
        NUM_CONV=2),
        ROI_BOX_HEAD=dict(
            CLS_AGNOSTIC_BBOX_REG=True,
        ),
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
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(60000, 80000),
            MAX_ITER=90000,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.02,
        ),
        IMS_PER_BATCH=16,
        IMS_PER_DEVICE=2,
        CHECKPOINT_PERIOD=30000,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(640, 672, 704, 736, 768, 800),
                      max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        )
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
