import logging
import numpy as np
import torch
import json

from cvpods.structures import Boxes, BoxMode, pairwise_iou
from cvpods.utils import PathManager, comm, create_small_table, create_table_with_header
import os
import os.path as osp

import pycocotools.mask as mask_util
import itertools

from .evaluator import DatasetEvaluator


class YouTubeVISEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, dataset, meta, cfg, distributed, output_dir=None, dump=False):
        """
        Args:
            dataset_name(str): name of the datasets to be evaluated.
            dataset(Dataset): A ConcatDataset
            meta(SimpleNamespace): datasets metadata.
            cfg(CfgNode): cvpods Config instance
            distributed(bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir(str): optional, an output ditectory to dump all results predicted on the datasets.

            dump(bool): If true after the evaluation is complated, a Markdown file that records
                the model evaluation metrics and corrspongding scores will be generated in the working directory.
        """
        self._dataset_name = dataset_name
        self._dataset = dataset.datasets[0]
        self._dump = dump
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = meta

    def reset(self):
        self.predictions = []
        self.results_for_vis = []

    def process(self, inputs, outputs):
        """
        Args:
            input: the inputs to a youtube vis model (e.g., MaskTrackRCNN).
                It is a list of dict. Each dict corresponds to an image and contains
                keys like "height", "width", "file_name", "image_id".
            output: the outputs of a youtube vis model. It is a list of dicts with the key
                "instances" that contains :class: `Instance`
        """
        for input, output in zip(inputs, outputs):
            prediction = {}

            prediction["is_first"] = input["is_first"]
            prediction["video_idx"] = input["video_idx"]
            prediction["frame_id"] = input["frame_id"]
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction_vis = {}
                prediction_vis["instances"] = instances_to_youtubevis_json(instances,
                                                                           video_id=input["video_idx"],
                                                                           frame_id=input["frame_id"],
                                                                           is_first=input["is_first"])          # list[dict] each dict for one object in one image
                prediction["instances"] = instances

            self.predictions.append(prediction)
            self.results_for_vis.append(prediction_vis)

            del prediction
            del prediction_vis

    def evaluate(self):
        if len(self.predictions) == 0:
            self._logger.warning("[YouTubeVISEvaluator] Did not receive any valid predictions.")
            return {}

        self._logger.info("Saving results for visualizing")
        self.save_results_for_visualize()
        self._logger.info("Saving results for visualizing done")
        json_results = []
        vid_objs = {}
        for idx in range(len(self._dataset)):
            vid_id, frame_id = self._dataset.valid_img_ids[idx]
            if idx == len(self._dataset) - 1:
                is_last = True
            else:
                _, frame_id_next = self._dataset.valid_img_ids[idx+1]
                is_last = frame_id_next == 0
            prediction = self.predictions[idx]
            instances = prediction["instances"]
            pred_boxes = instances.pred_boxes.tensor
            scores = instances.scores
            pred_classes = instances.pred_classes
            pred_masks = instances.pred_masks
            pred_obj_ids = instances.pred_obj_ids
            for pred_box, score, pred_class, pred_mask, pred_obj_id in zip(
                    pred_boxes, scores, pred_classes, pred_masks, pred_obj_ids):
                pred_obj_id = pred_obj_id.item()
                if pred_obj_id < 0:
                    continue
                if pred_obj_id not in vid_objs:
                    vid_objs[pred_obj_id] = {'scores':[], 'cats':[], 'segms':{}}
                vid_objs[pred_obj_id]['scores'].append(score)
                vid_objs[pred_obj_id]['cats'].append(pred_class)
                pred_mask = pred_mask.byte()
                pred_mask = np.array(pred_mask[:, :, np.newaxis], order='F').astype(np.uint8)
                rle = mask_util.encode(pred_mask)[0]
                rle['counts'] = rle['counts'].decode()
                vid_objs[pred_obj_id]['segms'][frame_id] = rle
            if is_last:
                for obj_id, obj in vid_objs.items():
                    data = dict()
                    data['video_id'] = vid_id + 1
                    data['score'] = np.array(obj['scores']).mean().item()
                    data['category_id'] = np.bincount(np.array(obj['cats'])).argmax().item() + 1
                    vid_seg = []
                    for fid in range(frame_id+1):
                        if fid in obj['segms']:
                            vid_seg.append(obj['segms'][fid])
                        else:
                            vid_seg.append(None)
                    data['segmentations'] = vid_seg
                    json_results.append(data)
                vid_objs = {}
        self._output_dir += '/results.json'
        with open(self._output_dir, 'w') as f:
            json.dump(json_results, f)

    def save_results_for_visualize(self):
        if len(self.predictions) == 0:
            self._logger.warning("[YouTubeVISEvaluator] Dis not receive ang valid predictions")
            return
        self._youtube_results = list(itertools.chain(*[x["instances"] for x in self.results_for_vis]))      # list of dicts all objects detected by all the images

        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v:k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._youtube_results:
                category_id = result["category_id"]
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            os.makedirs(self._output_dir)
            file_path = osp.join(self._output_dir, "youtubevis_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._youtube_results))
                f.flush()

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox", )
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm", )
        if cfg.MODEL.TRACK_ON:
            tasks = tasks + ("track", )
        return tasks

def instances_to_youtubevis_json(instances, video_id, frame_id, is_first):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.
    Args:
        instances (Instances):
        video_id (int): the video id
        frame_id (int): the frame if one video
        is_first (bool): whether this image is the first image of a video
    Returns:
        list[dict]: list of json annotations in COCO format. each dict is for one instance detected by model
    """
    num_instance = len(instances)                   # number instances detected in this image
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()                 # num_boxes, 4
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)          # convert boxes from xyxy mode to xyhw mode
    boxes = boxes.tolist()                                      # list of list of float len1=num boxes, len2=4
    scores = instances.scores.tolist()                          # len=num_boxes
    classes = instances.pred_classes.tolist()                   # len=num_boxes
    obj_ids = instances.pred_obj_ids.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire datasets
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]                           # list of dict
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    results = []
    for k in range(num_instance):                       # Loop for each objects in one image
        if obj_ids[k] < 0:
            continue
        result = {
            "video_id": video_id,
            "frame_id": frame_id,
            "is_first": is_first,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
            "obj_id": obj_ids[k]
        }
        if has_mask:
            result["segmentation"] = rles[k]
        results.append(result)
    return results                  # list of dict
