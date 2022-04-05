import os.path as osp

from cvpods.configs.pointrend_config import PointRendRCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="/mnt/lustreold/share_data/lixiangtai/pretrained/R-101.pkl",
        MASK_ON=True,
        RESNETS=dict(DEPTH=101),
        ROI_HEADS=dict(NUM_CLASSES=8),
        POINT_HEAD=dict(
            NUM_CLASSES=8,
        ),
    ),
    DATASETS=dict(
        TRAIN=("cityscapes_fine_instance_seg_train",),
        TEST=("cityscapes_fine_instance_seg_test",),
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


class PointRendRCNNConfig(PointRendRCNNFPNConfig):
    def __init__(self):
        super(PointRendRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = PointRendRCNNConfig()
