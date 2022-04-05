#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   coco.py
@Author             :   Benjin Zhu Xiangtai Li
'''

import contextlib
import copy
import datetime
import io
import json
import logging
import os
import os.path as osp

import numpy as np
import torch

from PIL import Image

from cvpods.structures import Boxes, BoxMode, PolygonMasks
from cvpods.utils import PathManager, Timer, file_lock

from pycocotools import mask as coco_mask

from ..registry import DATASETS
from ..base_dataset import BaseDataset
from ..detection_utils import (annotations_to_instances, check_image_size,
                               create_keypoint_hflip_indices,
                               filter_empty_instances, read_image)
from .builtin_meta import _get_builtin_metadata
from .paths_route import _PREDEFINED_SPLITS_COCO

"""
This file contains functions to parse COCO-format annotations into dicts in "cvpods format".
"""

logger = logging.getLogger(__name__)


@DATASETS.register()
class COCODataset(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(COCODataset, self).__init__(cfg, dataset_name, transforms, is_train)

        if "panoptic" in dataset_name:
            self.task_key = "panoptic"      # for task: panoptic/semantic segmentation
        elif "keypoints" in dataset_name:
            self.task_key = "coco_person"   # for task: keypoints detection
        else:
            self.task_key = "coco"          # for task: instance detection/segmentation

        self.meta = self._get_metadata()
        if self.task_key in ["coco", "coco_person"]:
            self.dataset_dicts = self._load_annotations(
                self.meta["json_file"],
                self.meta["image_root"],
                dataset_name)
        elif self.task_key in ["panoptic"]:
            # panoptic segmentation task, support below datasets names:
            #  * coco_2017_train_panoptic_separated
            #  * coco_2017_val_panoptic_separated
            #  * coco_2017_val_100_panoptic_separated
            if "_separated" in dataset_name:
                self.dataset_dicts = self._load_annotations(
                    self.meta["json_file"],
                    self.meta["image_root"],
                    dataset_name,
                    ["sem_seg"]
                )
            # semantic segmentation task, support below datasets names:
            #  * coco_2017_train_panoptic_stuffonly
            #  * coco_2017_val_panoptic_stuffonly
            #  * coco_2017_val_100_panoptic_stuffonly
            elif "_stuffonly" in dataset_name:
                self.dataset_dicts = load_sem_seg(
                    self.meta["sem_seg_root"],
                    self.meta["image_root"],
                )
            else:
                raise ValueError(f"Unknow datasets name: {self.name}.")

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

        self.class_splits = {}
        extra_args = cfg.DATASETS.DATASET_ARGS
        if extra_args.LOAD_EMBEDDINGS:
            self.class_embeddings = {}
            with open(self.meta["json_file"], 'r') as fin:
                ann_data = json.load(fin)
                for item in ann_data['categories']:
                    emb = item['embedding'][extra_args['EMB_KEY']]
                    self.class_embeddings[item['id']] = np.asarray(emb, dtype=np.float32)
                    if 'split' in item:
                        if item['split'] not in self.class_splits:
                            self.class_splits[item['split']] = []
                        self.class_splits[item['split']].append(item['id'])
            self.class_emb_mtx = np.zeros(
                (len(self.contiguous_category_id_to_json_id) + 1, extra_args['EMB_DIM']),  # (n+1, d)
                dtype=np.float32)
            for i, cid in self.contiguous_category_id_to_json_id.items():
                self.class_emb_mtx[i, :] = self.class_embeddings[cid]
            logger.info("Build Class Embeddings Matrix Done")


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

        if "sem_seg_file_name" in dataset_dict:
            # assert annotations is None
            # annotations = []
            with PathManager.open(dataset_dict.get("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            annotations.append({"sem_seg": sem_seg_gt})

        # apply transfrom on images and annotations
        image, annotations = self._apply_transforms(
            image, annotations)

        if "sem_seg_file_name" in dataset_dict:
            dataset_dict.pop("sem_seg_file_name")
            sem_seg_gt = annotations[-1].pop("sem_seg")
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt

        if annotations is not None:
            image_shape = image.shape[:2]  # h, w
            # a little hacky to ignore the last dimension of annotation for panoptic segmentation
            # remove
            if "sem_seg" in dataset_dict:
                instances = annotations_to_instances(
                    annotations[:-1], image_shape, mask_format=self.mask_format
                )
            else:
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

    def _load_annotations(self,
                          json_file,
                          image_root,
                          dataset_name=None,
                          extra_annotation_keys=None):
        """
        Load a json file with COCO's instances annotation format.
        Currently supports instance detection, instance segmentation,
        and person keypoints annotations.

        Args:
            json_file (str): full path to the json file in COCO instances annotation format.
            image_root (str): the directory where the images in this json file exists.
            dataset_name (str): the name of the datasets (e.g., coco_2017_train).
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
        from pycocotools.coco import COCO

        timer = Timer()
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f} seconds.".format(
                json_file, timer.seconds()))

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(coco_api.getCatIds())  # (0, n-1) : ids
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()  # ids : (0, n-1)
        }

        id_map = None
        if dataset_name is not None:
            meta = self.meta
            cat_ids = sorted(coco_api.getCatIds())
            cats = coco_api.loadCats(cat_ids)
            # The categories in a custom json file may not be sorted.
            thing_classes = [
                c["name"] for c in sorted(cats, key=lambda x: x["id"])
            ]
            meta["thing_classes"] = thing_classes

            # In COCO, certain category ids are artificially removed,
            # and by convention they are always ignored.
            # We deal with COCO's id issue and translate
            # the category ids to contiguous ids in [0, 80).

            # It works by looking at the "categories" field in the json, therefore
            # if users' own json also have incontiguous ids, we'll
            # apply this mapping as well but print a warning.
            if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
                if "coco" not in dataset_name:
                    logger.warning("""
    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
    """)
            id_map = {v: i for i, v in enumerate(cat_ids)}
            meta["thing_dataset_id_to_contiguous_id"] = id_map

        # sort indices for reproducible results
        img_ids = sorted(coco_api.imgs.keys())
        # imgs is a list of dicts, each looks something like:
        # {'license': 4,
        #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
        #  'file_name': 'COCO_val2014_000000001268.jpg',
        #  'height': 427,
        #  'width': 640,
        #  'date_captured': '2013-11-17 05:57:24',
        #  'id': 1268}
        imgs = coco_api.loadImgs(img_ids)
        # anns is a list[list[dict]], where each dict is an annotation
        # record for an object. The inner list enumerates the objects in an image
        # and the outer list enumerates over images. Example of anns[0]:
        # [{'segmentation': [[192.81,
        #     247.09,
        #     ...
        #     219.03,
        #     249.06]],
        #   'area': 1035.749,
        #   'iscrowd': 0,
        #   'image_id': 1268,
        #   'bbox': [192.81, 224.8, 74.73, 33.43],
        #   'category_id': 16,
        #   'id': 42986},
        #  ...]
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

        if "minival" not in json_file:
            # The popular valminusminival & minival annotations for COCO2014 contain this bug.
            # However the ratio of buggy annotations there is tiny and does not affect accuracy.
            # Therefore we explicitly white-list them.
            ann_ids = [
                ann["id"] for anns_per_image in anns for ann in anns_per_image
            ]
            assert len(set(ann_ids)) == len(
                ann_ids), "Annotation ids in '{}' are not unique!".format(
                    json_file)

        imgs_anns = list(zip(imgs, anns))

        logger.info("Loaded {} images in COCO format from {}".format(
            len(imgs_anns), json_file))

        dataset_dicts = []

        ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"
                    ] + (extra_annotation_keys or [])

        num_instances_without_valid_segmentation = 0

        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(image_root,
                                               img_dict["file_name"])
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                # Check that the image_id in this annotation is the same as
                # the image_id we're looking at.
                # This fails only when the data parsing logic or the annotation file is buggy.

                # The original COCO valminusminival2014 & minival2014 annotation files
                # actually contains bugs that, together with certain ways of using COCO API,
                # can trigger this assertion.
                assert anno["image_id"] == image_id
                assert anno.get("ignore", 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}

                segm = anno.get("segmentation", None)
                if segm:    # either list[list[float]] or dict(RLE)
                    if not isinstance(segm, dict):
                        # filter out invalid polygons (< 3 points)
                        segm = [
                            poly for poly in segm
                            if len(poly) % 2 == 0 and len(poly) >= 6
                        ]
                        if len(segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue    # ignore this instance
                    obj["segmentation"] = segm

                keypts = anno.get("keypoints", None)
                if keypts:    # list[int]
                    for idx, v in enumerate(keypts):
                        if idx % 3 != 2:
                            # COCO's segmentation coordinates are floating points in [0, H or W],
                            # but keypoint coordinates are integers in [0, H-1 or W-1]
                            # Therefore we assume the coordinates are "pixel indices" and
                            # add 0.5 to convert to floating point coordinates.
                            keypts[idx] = v + 0.5
                    obj["keypoints"] = keypts

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                objs.append(obj)
            if extra_annotation_keys is not None and "sem_seg" in extra_annotation_keys:
                record['sem_seg_file_name'] = os.path.join(self.meta['sem_seg_root'],
                                               img_dict["file_name"].split(".")[0]+".png")

            record["annotations"] = objs

            dataset_dicts.append(record)

        if num_instances_without_valid_segmentation > 0:
            logger.warning(
                "Filtered out {} instances without valid segmentation. "
                "There might be issues in your datasets generation process.".
                format(num_instances_without_valid_segmentation))

        return dataset_dicts

    def _get_metadata(self):
        if self.task_key in ["coco", "coco_person"]:
            meta = _get_builtin_metadata("coco")
            image_root, json_file = _PREDEFINED_SPLITS_COCO[self.task_key][self.name]
            meta["image_root"] = osp.join(self.data_root, image_root) \
                if "://" not in image_root else image_root
            meta["json_file"] = osp.join(self.data_root, json_file) \
                if "://" not in image_root else osp.join(image_root, json_file)
            meta["evaluator_type"] = _PREDEFINED_SPLITS_COCO["evaluator_type"][self.task_key]
        elif self.task_key in ["panoptic"]:
            meta = _get_builtin_metadata("coco_panoptic_separated")
            prefix_instances = self.name[: -len("_panoptic_separated")]
            prefix_panoptic = self.name[: -len("_separated")]
            eval_key = self.name[-len("panoptic_separated"):]

            image_root, json_file = _PREDEFINED_SPLITS_COCO["coco"][prefix_instances]
            panoptic_root, panoptic_json, semantic_root = \
                _PREDEFINED_SPLITS_COCO[self.task_key][prefix_panoptic]
            meta["image_root"] = osp.join(self.data_root, image_root) \
                if "://" not in image_root else image_root
            meta["sem_seg_root"] = os.path.join(self.data_root, semantic_root) \
                if "://" not in semantic_root else semantic_root
            meta["evaluator_type"] = _PREDEFINED_SPLITS_COCO["evaluator_type"][eval_key]

            if "_separated" in self.name:
                meta["json_file"] = osp.join(self.data_root, json_file) \
                    if "://" not in image_root else osp.join(image_root, json_file)
                meta["panoptic_root"] = os.path.join(self.data_root, panoptic_root) \
                    if "://" not in panoptic_root else panoptic_root
                meta["panoptic_json"] = os.path.join(self.data_root, panoptic_json) \
                    if "://" not in panoptic_root else osp.join(panoptic_root, panoptic_json)
        return meta

    def evaluate(self, predictions):
        """Dataset must provide a evaluation function to evaluate model."""
        raise NotImplementedError

    @property
    def ground_truth_annotations(self):
        return self.dataset_dicts


# TODO this function is not specific to COCO, except for the "image_id" logic.
def load_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg"):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in cvpods standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(
            os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f)
         for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_root, f)
         for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    if len(input_files) != len(gt_files):
        logger.warn(
            "Directory {} and {} has {} and {} files, respectively.".format(
                image_root, gt_root, len(input_files), len(gt_files)))
        input_basenames = [
            os.path.basename(f)[:-len(image_ext)] for f in input_files
        ]
        gt_basenames = [os.path.basename(f)[:-len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        # sort, otherwise each worker may obtain a list[dict] in different order
        intersect = sorted(intersect)
        logger.warn("Will use their intersection of {} files.".format(
            len(intersect)))
        input_files = [
            os.path.join(image_root, f + image_ext) for f in intersect
        ]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]

    logger.info("Loaded {} images with semantic segmentation from {}".format(
        len(input_files), image_root))

    dataset_dicts = []
    for (img_path, gt_path) in zip(input_files, gt_files):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts


def convert_to_coco_dict(dataset_name, dataset_dicts, metadata):
    """
    Convert a datasets in detectron2's standard format into COCO json format
    Generic datasets description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset
    COCO data format description can be found here:
    http://cocodataset.org/#format-data
    Args:
        dataset_name:
            name of the source datasets
            must be registered in DatastCatalog and in detectron2's standard format
    Returns:
        coco_dict: serializable dict in COCO json format
    """
    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {
            v: k
            for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
        }

        def reverse_id_mapper(contiguous_id): return reverse_id_mapping[contiguous_id]  # noqa
    else:
        def reverse_id_mapper(contiguous_id): return contiguous_id    # noqa

    categories = [{
        "id": reverse_id_mapper(id),
        "name": name
    } for id, name in enumerate(metadata.thing_classes)]

    logger.info("Converting datasets dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": image_dict["width"],
            "height": image_dict["height"],
            "file_name": image_dict["file_name"],
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict["annotations"]
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format
            bbox = annotation["bbox"]
            bbox_mode = annotation["bbox_mode"]
            bbox = BoxMode.convert(bbox, bbox_mode, BoxMode.XYWH_ABS)

            # COCO requirement: instance area
            if "segmentation" in annotation:
                # Computing areas for instances by counting the pixels
                segmentation = annotation["segmentation"]
                # TODO: check segmentation type: RLE, BinaryMask or Polygon
                polygons = PolygonMasks([segmentation])
                area = polygons.area()[0].item()
            else:
                # Computing areas using bounding boxes
                bbox_xy = BoxMode.convert(bbox, BoxMode.XYWH_ABS,
                                          BoxMode.XYXY_ABS)
                area = Boxes([bbox_xy]).area()[0].item()

            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]    # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    num_keypoints = annotation["num_keypoints"]
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = area
            coco_annotation["category_id"] = reverse_id_mapper(
                annotation["category_id"])
            coco_annotation["iscrowd"] = annotation.get("iscrowd", 0)

            # Add optional fields
            if "keypoints" in annotation:
                coco_annotation["keypoints"] = keypoints
                coco_annotation["num_keypoints"] = num_keypoints

            if "segmentation" in annotation:
                coco_annotation["segmentation"] = annotation["segmentation"]

            coco_annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"num images: {len(coco_images)}, num annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description":
        "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "licenses": None,
    }
    return coco_dict


def convert_to_coco_json(dataset_name, output_file, allow_cached=True):
    """
    Converts datasets into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.
    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The datasets or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.info(
                f"Cached annotations in COCO format already exist: {output_file}"
            )
        else:
            logger.info(
                f"Converting datasets annotations in '{dataset_name}' to COCO format ...)"
            )
            coco_dict = convert_to_coco_dict(dataset_name)

            with PathManager.open(output_file, "w") as json_file:
                logger.info(
                    f"Caching annotations in COCO format: {output_file}")
                json.dump(coco_dict, json_file)


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks