# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved

from .cityscapes import CityScapesDataset
from .coco import COCODataset
from .imagenet import ImageNetDataset
from .voc import VOCDataset
from .widerface import WiderFaceDataset
from .lvis import LVISDataset
from .citypersons import CityPersonsDataset
from .crowdhuman import CrowdHumanDataset
from .youtubevis import YTVisDataset
from .ovis import OVisDataset
from .coco_captions import COCOCaptionsDataset

__all__ = [
    "COCODataset",
    "VOCDataset",
    "CityScapesDataset",
    "ImageNetDataset",
    "WiderFaceDataset",
    "LVISDataset",
    "CityPersonsDataset",
    "CrowdHumanDataset",
    "YTVisDataset",
    "OVisDataset",
    "COCOCaptionsDataset"
]
