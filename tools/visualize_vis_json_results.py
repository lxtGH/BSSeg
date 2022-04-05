#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import os
from collections import defaultdict
import tqdm
import pycocotools.mask as mask_util

import cv2
import numpy as np

from cvpods.data import build_dataset
from cvpods.structures import Boxes, BoxMode, Instances
from cvpods.utils import PathManager, Visualizer, dynamic_import, setup_logger, ColorMode


def setup_cfg(path, logger):
    # load config from file and command-line arguments
    assert path.endswith(".py")
    path, module = os.path.split(path)
    module = module.rstrip(".py")
    cfg = dynamic_import(module, path).config
    if cfg.DATASETS.CUSTOM_TYPE != ["ConcatDataset", dict()]:
        logger.warning("Ignore cfg.DATASETS.CUSTOM_TYPE: {}. "
                       "Using (\"ConcatDataset\", dict())".format(cfg.DATASETS.CUSTOM_TYPE))
    cfg.DATASETS.CUSTOM_TYPE = ["ConcatDataset", dict()]

    return cfg


def create_instances(predictions, image_size):
    """
    Args:
        predictions: list(dict) len=num_instances in one image, each dict is for one instance
        with field keys: video_id, frame_id, is_first, is_last, category_id, bbox, score, obj_id
        image_size: tuple (len=2), height, width
    Returns:
        Instance: all instances in one image with field keys: scores(np.ndarray)[num_instances, ],
        pred_boxes(Boxes)[num_instances, 4], pred_classes(np.ndarray)[num_instances, ],
        obj_ids(np.ndarray)[num_instanes, ], is_first (bool) whether this image is the first frame of according video,
        is_last(bool) whether this image is the last frame of according video,
        pred_masks (list[dict(rle)])
    """
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])       # list(float)
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen])         # list of list of float len1=num_boxes, len2=4

    if score.shape[0] == 0:
        bbox = np.zeros((0, 4))
    else:
        bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)    # convert xyhw to x0y0x1y1

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])        # convert things id to train id shape = len(chosen)
    obj_ids = np.asarray([predictions[i]["obj_id"] for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels
    ret.obj_ids = obj_ids

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret

def gen_dummy_instance(image_size):

    ret = Instances(image_size)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--config", required=True,
                        help="path to a python file with a definition of `config`")
    parser.add_argument("--dataset",
                        help="name of the dataset. Use DATASETS.TEST[0] if not specified.",
                        default="")
    parser.add_argument("--vis_gt", action='store_true', help="Whether to visualize ground truth")
    parser.add_argument("--train_mode", action='store_true', help="Whether using training mode when building dataset")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()
    cfg = setup_cfg(args.config, logger)
    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)              # list[dicts]: all objects detected by all the images

    pred_by_image = defaultdict(dict)           # Loop for all detected objects then collate all objects for each image
    for p in predictions:
        if p["frame_id"] not in pred_by_image[p["video_id"]]:
            pred_by_image[p["video_id"]][p["frame_id"]] = []
        pred_by_image[p["video_id"]][p["frame_id"]].append(p)

    # TODO: add DatasetCatalog, MetadataCatalog
    if args.train_mode:
        dataset = build_dataset(
            cfg,
            [args.dataset] if args.dataset else [cfg.DATASETS.TEST[0]],
            transforms=[],
            is_train=True)
    else:
        dataset = build_dataset(
            cfg,
            [args.dataset] if args.dataset else [cfg.DATASETS.TEST[0]],
            transforms=[],
            is_train=False)
    dicts = dataset.datasets[0].dataset_dicts
    metadata = dataset.meta
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)

    for dic in tqdm.tqdm(dicts):                # Loop for each image
        if dic["video_idx"] == 43 or dic["video_idx"] == 62 or dic["video_idx"] == 263:
            is_first = dic["is_first"]
            if is_first:
                prev_colors = None
            video_file = os.path.join(args.output, str(dic["video_idx"]))
            if is_first:
                os.makedirs(video_file, exist_ok=True)
            img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
            basename = os.path.basename(dic["file_name"])       # image name

            if dic["frame_id"] not in pred_by_image[dic["video_idx"]].keys():
                predictions = gen_dummy_instance(image_size=img.shape[:2])
            else:
                predictions = create_instances(pred_by_image[dic["video_idx"]][dic["frame_id"]], img.shape[:2])      # Instance with field keys: scores, pred_classes, pred_boxes, pred_masks(optional) etc
            vis = Visualizer(img, metadata, instance_mode=ColorMode.VIDEO_SEGMENTATION)         # visualize one image
            if len(predictions) == 0:                   # no valid object for this image's visualize
                vis_pred = vis.output.get_image()
            else:
                vis_pred, prev_colors = vis.draw_instance_predictions_video_instance_segmentation(predictions, prev_colors=prev_colors)
                vis_pred = vis_pred.get_image()             # ndarray [H, W, 3]

            if args.vis_gt:
                vis = Visualizer(img, metadata)
                vis_gt = vis.draw_dataset_dict_vis(dic).get_image()
                concat = np.concatenate((vis_pred, vis_gt), axis=1)
                cv2.imwrite(os.path.join(video_file, basename), concat[:, :, ::-1])
            else:
                cv2.imwrite(os.path.join(video_file, basename), vis_pred[:, :, ::-1])
        else:
            continue