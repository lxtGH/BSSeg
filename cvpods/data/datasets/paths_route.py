#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   paths_route.py
@Time               :   2020/05/07 23:48:49
@Author             :   Benjin Zhu
@Contact            :   poodarchu@gmail.com
@Last Modified by   :   Xiangtai Li
@Last Modified time :   2021/2/1 13:24:39
'''

"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the datasets annotations

We hard-code some paths to the datasets that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new datasets / metadata for new datasets.
To add new datasets, refer to the tutorial "docs/DATASETS.md".
"""

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["dataset_type"] = "COCODataset"
_PREDEFINED_SPLITS_COCO["evaluator_type"] = {
    "coco": "coco",
    "coco_person": "coco",
    "panoptic_separated": "coco_panoptic_seg",
    "panoptic_stuffonly": "sem_seg",
}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train":
    ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val":
    ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival":
    ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100":
    ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017",
                        "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017",
                      "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017",
                       "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017",
                           "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017",
                          "coco/annotations/instances_val2017_100.json"),

    "coco_zeroshot_train": ("coco/train2017",
                           "coco/zero-shot/instances_train2017_seen_2.json"),

    "coco_not_zeroshot_val": ("coco/val2017",
                             "coco/zero-shot/instances_val2017_seen_2.json"
    ),
    "coco_zeroshot_val": ("coco/val2017",
                          "coco/zero-shot/instances_val2017_unseen_2.json"
    ),
    "coco_generalized_zeroshot_val": ("coco/val2017",
                                      "coco/zero-shot/instances_val2017_all_2.json"
    )
}

_PREDEFINED_SPLITS_COCOCAPTIONS = {
    "dataset_type": "COCOCaptionsDataset",
    "cococaption":{
    "cococaptions_train": ("coco/train2017",
                        "coco/annotations/captions_train2017.json"),
    "cococaptions_val": ("coco/val2017",
                      "coco/annotations/captions_val2017.json"),
    }
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "coco_keypoints_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "coco_keypoints_2014_val":
    ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "coco_keypoints_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "coco_keypoints_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "coco_keypoints_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014_100.json",
    ),
    "coco_keypoints_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "coco_keypoints_2017_val":
    ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "coco_keypoints_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}

_PREDEFINED_SPLITS_COCO["panoptic"] = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at cvpods/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}

# ==== Predefined datasets and splits for LVIS ==========

_PREDEFINED_SPLITS_LVIS = {
    "dataset_type": "LVISDataset",
    "evaluator_type": {
        "lvis_v0.5": "lvis",
        "lvis_v0.5_cocofied": "lvis",
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/train2017", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/val2017", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/val2017", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/test2017", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/train2017", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/val2017", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}

# ==== Predefined splits for raw cityscapes images ===========

_PREDEFINED_SPLITS_CITYSCAPES = {
    "dataset_type": "CityScapesDataset",
    "evaluator_type": {
        "cityscapes_instance": "cityscapes",
        "cityscapes_sem": "sem_seg",
    },
    "cityscapes": {
        "cityscapes_fine_instance_seg_train":
        ("cityscapes/leftImg8bit/train", "cityscapes/gtFine/train"),
        "cityscapes_fine_instance_seg_val":
        ("cityscapes/leftImg8bit/val", "cityscapes/gtFine/val"),
        "cityscapes_fine_instance_seg_test":
        ("cityscapes/leftImg8bit/test", "cityscapes/gtFine/test"),

        "cityscapes_fine_sem_seg_train":
        ("cityscapes/leftImg8bit/train", "cityscapes/gtFine/train"),
        "cityscapes_fine_sem_seg_val":
        ("cityscapes/leftImg8bit/val", "cityscapes/gtFine/val"),
        "cityscapes_fine_sem_seg_test":
        ("cityscapes/leftImg8bit/test", "cityscapes/gtFine/test"),

        "cityscapes_fine_panoptic-deeplab_seg_train":
        ("cityscapes/leftImg8bit/train", "cityscapes/gtFine/train"),
        "cityscapes_fine_panoptic-deeplab_seg_val":
        ("cityscapes/leftImg8bit/val", "cityscapes/gtFine/val"),
        "cityscapes_fine_panoptic-deeplab_seg_test":
        ("cityscapes/leftImg8bit/test", "cityscapes/gtFine/test"),

    },
}

# ==== Predefined splits for PASCAL VOC ===========

_PREDEFINED_SPLITS_VOC = {
    "dataset_type": "VOCDataset",
    "evaluator_type": {
        # "voc_2007": "pascal_voc",
        # "voc_2012": "pascal_voc",
        "voc": "pascal_voc",
    },
    "voc": {
        "voc_2007_trainval": ("voc/VOC2007", "trainval"),
        "voc_2007_train": ("voc/VOC2007", "train"),
        "voc_2007_val": ("voc/VOC2007", "val"),
        "voc_2007_test": ("voc/VOC2007", "test"),
        "voc_2012_trainval": ("voc/VOC2012", "trainval"),
        "voc_2012_train": ("voc/VOC2012", "train"),
        "voc_2012_val": ("voc/VOC2012", "val"),
    },
}

# ==== Predefined splits for citypersons ===========

_PREDEFINED_SPLITS_CITYPERSONS = {
    "dataset_type": "CityPersonsDataset",
    "evaluator_type": {
        "citypersons": "citypersons",
    },
    "citypersons": {
        "citypersons_train":
        ("citypersons/train", "citypersons/annotations/train.json"),
        "citypersons_val":
        ("citypersons/val", "citypersons/annotations/val.json"),
    }
}

# ==== Predefined splits for Objects365 ===========

_PREDEFINED_SPLITS_OBJECTS365 = {
    "dataset_type": "Objects365Dataset",
    "evaluator_type": {
        "objects365": "coco",
    },
    "objects365": {
        "objects365_train":
        ("objects365/train",
         "objects365/annotations/objects365_train_20190423.json"),
        "objects365_val":
        ("objects365/val",
         "objects365/annotations/objects365_val_20190423.json"),
        "objects365_test":
        ("objects365/test",
         "objects365/annotations/objects365_test_20190423.json"),
        "objects365_tiny_train":
        ("objects365/train",
         "objects365/annotations/objects365_Tiny_train.json"),
        "objects365_tiny_val":
        ("objects365/val", "objects365/annotations/objects365_Tiny_val.json"),
    }
}

# ==== Predefined splits for WiderFace ===========

_PREDEFINED_SPLITS_WIDERFACE = {
    "dataset_type": "WiderFaceDataset",
    "evaluator_type": {
        "widerface_2019": "widerface",
    },
    "widerface_2019": {
        "widerface_2019_train":
        ("widerface/train",
         "widerface/annotations/widerface2019_train_cocostyle.json"),
        "widerface_2019_val":
        ("widerface/val",
         "widerface/annotations/widerface2019_val_cocostyle.json"),
    }
}

# ==== Predefined datasets and splits for ImageNet ==========

_PREDEFINED_SPLITS_IMAGENET = {
    "dataset_type": "ImageNetDataset",
    "evaluator_type": {
        "imagenet": "classification"
    },
    "imagenet": {
        "imagenet_train": ("imagenet", "train"),
        "imagenet_val": ("imagenet", "val"),
    }
}


# ==== Predefined splits for CrowdHuman ===========

_PREDEFINED_SPLITS_CROWDHUMAN = {
    "dataset_type": "CrowdHumanDataset",
    "evaluator_type": {
        "crowdhuman": "crowdhuman",
    },
    "crowdhuman": {
        "crowdhuman_train":
        ("crowdhuman/Images",
         "crowdhuman/annotations/annotation_train.odgt"),
        "crowdhuman_val":
        ("crowdhuman/Images",
         "crowdhuman/annotations/annotation_val.odgt"),
    }
}


# ====Predefined splits for YoutubeVis ===============

_PREDEFINED_SPLITS_YOUTUBEVIS = {
    "dataset_type": "YTVisDataset",
    "evaluator_type": {
        "youtubevis": "youtubevis"
    },
    "youtubevis": {
        "youtubevis_train":
        ("youtubevis/train/JPEGImages",
         "youtubevis/annotations/train.json"),
        "youtubevis_valid":
        ("youtubevis/valid/JPEGImages",
         "youtubevis/annotations/valid.json")
    }
}

# ====Predefined splits for YoutubeVisWithLimitSampleRange ===============

_PREDEFINED_SPLITS_YOUTUBEVISLIMIT = {
    "dataset_type": "YTVisDatasetLimit",
    "evaluator_type": {
        "youtubevis": "youtubevis"
    },
    "youtubevislimit": {
        "youtubevislimit_train":
        ("youtubevis/train/JPEGImages",
         "youtubevis/annotations/train.json"),
        "youtubevislimit_valid":
        ("youtubevis/valid/JPEGImages",
         "youtubevis/annotations/valid.json")
    }
}

# ====Predefined splits for Youtube2021VisWithLimitSampleRange ===============

_PREDEFINED_SPLITS_YOUTUBEVIS2021LIMIT = {
    "dataset_type": "YTVisDataset2021Limit",
    "evaluator_type": {
        "youtubevis": "youtubevis"
    },
    "youtubevis2021limit": {
        "youtubevis2021limit_train":
        ("youtubevis2021/train/JPEGImages",
         "youtubevis2021/train/instances.json"),
        "youtubevis2021limit_valid":
        ("youtubevis2021/valid/JPEGImages",
         "youtubevis2021/valid/instances.json")
    }
}



# ====Predefined splits for OVis ===============

_PREDEFINED_SPLITS_OVIS = {
    "dataset_type": "OVisDataset",
    "evaluator_type": {
        "ovis": "youtubevis" # hack
    },
    "ovis": {
        "ovis_train":
        ("ovis/Images",
         "ovis/annotations_train.json"),
        "ovis_valid":
        ("ovis/Images",
         "ovis/annotations_valid.json")
    }

}

_PREDEFINED_SPLITS_OVISLIMIT = {
    "dataset_type": "OVISDatasetLimit",
    "evaluator_type": {
        "ovis": "youtubevis" # hack
    },

    "ovislimit": {
        "ovislimit_train":
        ("ovis/Images",
         "ovis/annotations_train.json"),
        "ovislimit_valid":
        ("ovis/Images",
         "ovis/annotations_valid.json")
    }

}