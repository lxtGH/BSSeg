import copy
import json
import logging
import numpy as np
import os
import os.path as osp
import torch

from cvpods.structures import BoxMode
from cvpods.utils import PathManager, Timer

from ..registry import DATASETS
from ..base_dataset import BaseDataset
from ..detection_utils import (
    annotations_to_instances, check_image_size, create_keypoint_hflip_indices,
    filter_empty_instances, read_image)
from .paths_route import _PREDEFINED_SPLITS_CROWDHUMAN


"""
This file contains functions to parse COCO-format annotations into dicts in "cvpods format".
"""

logger = logging.getLogger(__name__)


@DATASETS.register()
class CrowdHumanDataset(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(CrowdHumanDataset, self).__init__(cfg, dataset_name, transforms, is_train)
        self.dataset_key = "_".join(self.name.split('_')[:-1])
        image_root, json_file = _PREDEFINED_SPLITS_CROWDHUMAN[self.dataset_key][self.name]
        self.json_file = osp.join(self.data_root, json_file) \
            if "://" not in image_root else osp.join(image_root, json_file)
        self.image_root = osp.join(self.data_root, image_root) \
            if "://" not in image_root else image_root

        self.meta = self._get_metadata()

        self.dataset_dicts = self._load_annotations(
            self.json_file,
            self.image_root,
            self.name,
            extra_annotation_keys=None)

        if is_train:
            self.dataset_dicts = self._filter_annotations()
            self._set_group_flag()

        self.eval_with_gt = cfg.TEST.get("WITH_GT", False)

        # fmt: off
        self.data_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on

        if self.keypoint_on:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

    def __getitem__(self, index):
        """Load data, apply transforms, converto to Instances.
        """
        dataset_dict = copy.deepcopy(self.dataset_dicts[index])

        # read image
        image = read_image(dataset_dict["file_name"], format=self.data_format)
        check_image_size(dataset_dict, image)

        if "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            annotations = [
                ann for ann in annotations if ann.get("iscrowd", 0) == 0]
        else:
            annotations = None

        # apply transfrom
        image, annotations = self._apply_transforms(
            image, annotations)

        if annotations is not None:
            image_shape = image.shape[:2]  # h, w

            instances = annotations_to_instances(
                annotations, image_shape, mask_format=self.mask_format
            )

            # # Create a tight bounding box from masks, useful when image is cropped
            # if self.crop_gen and instances.has("gt_masks"):
            #     instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            dataset_dict["instances"] = filter_empty_instances(instances)

        # convert to Instance type
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        # h, w, c -> c, h, w
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))

        return dataset_dict

    def __reset__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset_dicts)

    def _load_annotations(
        self, json_file, image_root,
        dataset_name=None, extra_annotation_keys=None
    ):
        """
        Load a json file with CrowdHuman's instances annotation format.
        Currently supports instance detection, instance segmentation,
        and person keypoints annotations.

        Args:
            json_file (str): full path to the json file in CrowdHuman instances annotation format.
            image_root (str): the directory where the images in this json file exists.
            dataset_name (str): the name of the datasets (e.g., CrowdHuman_train).
                If provided, this function will also put "thing_classes" into
                the metadata associated with this datasets.
            extra_annotation_keys (list[str]): list of per-annotation keys that should also be
                loaded into the datasets dict (besides "iscrowd", "bbox", "keypoints",
                "category_id", "segmentation"). The values for these keys will be returned as-is.
                For example, the densepose annotations are loaded in this way.

        Returns:
            list[dict]: a list of dicts in cvpods standard format. (See
            `Using Custom Datasets </tutorials/datasets.html>`_ )

        Notes:
            1. This function does not read the image files.
               The results do not have the "image" field.
        """
        timer = Timer()
        json_file = PathManager.get_local_path(json_file)
        with open(json_file, 'r') as file:
            gt_records = file.readlines()
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f} seconds.".format(
                json_file, timer.seconds()))

        logger.info("Loaded {} images in CrowdHuman format from {}".format(
            len(gt_records), json_file))

        dataset_dicts = []

        ann_keys = ["tag", "hbox", "vbox", "head_attr", "extra"]
        for anno_str in gt_records:
            anno_dict = json.loads(anno_str)

            record = {}
            record["file_name"] = os.path.join(image_root, "{}.jpg".format(anno_dict["ID"]))
            record["height"] = anno_dict["height"]
            record["width"] = anno_dict["width"]
            record["image_id"] = anno_dict["ID"]

            objs = []
            for anno in anno_dict['gtboxes']:
                # Check that the image_id in this annotation is the same as
                # the image_id we're looking at.
                # This fails only when the data parsing logic or the annotation file is buggy.

                # The original COCO valminusminival2014 & minival2014 annotation files
                # actually contains bugs that, together with certain ways of using COCO API,
                # can trigger this assertion.
                obj = {key: anno[key] for key in ann_keys if key in anno}
                obj["bbox"] = anno["fbox"]
                obj["category_id"] = 0

                if 'extra' in anno and 'ignore' in anno['extra'] and anno['extra']['ignore'] != 0:
                    obj["category_id"] = -1

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

        return dataset_dicts

    def _get_metadata(self):
        meta = {}
        meta["image_root"] = self.image_root
        meta["json_file"] = self.json_file
        meta["evaluator_type"] = _PREDEFINED_SPLITS_CROWDHUMAN["evaluator_type"][self.dataset_key]
        meta["thing_classes"] = ['person']

        return meta

    def evaluate(self, predictions):
        """Dataset must provide a evaluation function to evaluate model."""
        raise NotImplementedError

    @property
    def ground_truth_annotations(self):
        return self.dataset_dicts
