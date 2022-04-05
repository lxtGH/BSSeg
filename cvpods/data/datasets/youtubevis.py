import contextlib
import copy
import io
import logging
import random
import os.path as osp
import numpy as np
import torch

from collections import defaultdict
from cvpods.structures import BoxMode
from cvpods.utils import PathManager, Timer

from ..registry import DATASETS
from ..base_dataset import BaseDataset
from ..detection_utils import (annotations_to_instances_vis, check_image_size,
                               filter_empty_instances, read_image)
from .builtin_meta import _get_builtin_metadata
from .paths_route import _PREDEFINED_SPLITS_YOUTUBEVIS, _PREDEFINED_SPLITS_YOUTUBEVISLIMIT, \
    _PREDEFINED_SPLITS_YOUTUBEVIS2021LIMIT

from cvpods.data.datasets.ext.ytvos import YTVOS

logger = logging.getLogger(__name__)


@DATASETS.register()
class YTVisDataset(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(YTVisDataset, self).__init__(cfg, dataset_name, transforms, is_train)

        self.task_key = 'youtubevis'
        self.meta = self._get_metadata()
        self.train_mode = is_train
        self.valid_img_ids = self._load_annotations(
            self.meta["json_file"],
            self.meta["image_root"],
            dataset_name)
        self._check()

        if is_train:
            self._set_group_flag()

        self.eval_with_gt = cfg.TEST.get("WITH_GT", False)

        self.data_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self.track_on = cfg.MODEL.TRACK_ON

    def __len__(self):
        return len(self.valid_img_ids)

    def __getitem__(self, idx):
        if self.train_mode:
            return self._prepare_train_img(idx)
        else:
            return self._prepare_test_img(idx)

    def _sample_ref(self, idx):
        """
        Sample another frame in the same sequence as reference
        """
        vid, frame_id = idx  # 0-based
        vid_info = self.vid_infos[vid]
        sample_range = range(len(vid_info['file_names']))  # [0, len)
        valid_samples = []
        for i in sample_range:
            ref_idx = (vid, i)
            if i != frame_id and ref_idx in self.valid_img_ids:
                valid_samples.append(ref_idx)
        assert len(valid_samples) > 0
        return random.choice(valid_samples)

    def _prepare_train_img(self, idx):

        return_dict = {}
        query_vid, query_frame_id = self.valid_img_ids[idx]  # 0-based
        query_dataset_dict = copy.deepcopy(self.vid2img2info[query_vid][query_frame_id])  # 0-based
        reference_vid, reference_frame_id = self._sample_ref((query_vid, query_frame_id))  # 0-based
        reference_dataset_dict = copy.deepcopy(self.vid2img2info[reference_vid][reference_frame_id])

        query_img = read_image(query_dataset_dict["file_name"], format=self.data_format)
        check_image_size(query_dataset_dict, query_img)
        reference_img = read_image(reference_dataset_dict["file_name"], format=self.data_format)
        check_image_size(reference_dataset_dict, reference_img)

        if "annotations" in query_dataset_dict:
            query_annotations = query_dataset_dict.pop("annotations")  # list of dicts
        else:
            query_annotations = None
        if "annotations" in reference_dataset_dict:
            reference_annotations = reference_dataset_dict.pop("annotations")
        else:
            reference_annotations = None

        (query_img, reference_img), (query_annotations, reference_annotations) = \
            self._apply_transforms(images=[query_img, reference_img],
                                   annotationss=[query_annotations, reference_annotations])

        if query_annotations is not None:
            query_img_shape = query_img.shape[:2]
            query_instances = annotations_to_instances_vis(
                query_annotations, query_img_shape, mask_format=self.mask_format)
            query_dataset_dict["instances"] = filter_empty_instances(query_instances)

        if reference_annotations is not None:
            reference_img_shape = reference_img.shape[:2]
            reference_instances = annotations_to_instances_vis(
                reference_annotations, reference_img_shape, mask_format=self.mask_format)
            reference_dataset_dict["instances"] = filter_empty_instances(reference_instances)

        query_dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(query_img.transpose(2, 0, 1)))
        reference_dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(reference_img.transpose(2, 0, 1)))

        query_gt_ids = query_dataset_dict["instances"].instance_ids
        references_gt_ids = reference_dataset_dict["instances"].instance_ids

        gt_pids = [int((references_gt_ids == i).nonzero(as_tuple=False)) + 1 if i in references_gt_ids else 0 for i in
                   query_gt_ids]
        query_dataset_dict["instances"].gt_pids = torch.as_tensor(np.array(gt_pids))

        for k, v in query_dataset_dict.items():
            return_dict[k] = v
        for k, v in reference_dataset_dict.items():
            return_dict[k + "_reference"] = v
        del query_dataset_dict, reference_dataset_dict
        return return_dict

    def _prepare_test_img(self, idx):
        vid, frame_id = self.valid_img_ids[idx]  # 0-based
        dataset_dict = copy.deepcopy(self.vid2img2info[vid][frame_id])

        image = read_image(dataset_dict["file_name"], format=self.data_format)
        check_image_size(dataset_dict, image)

        image = self._apply_transforms(images=[image], annotationss=[None])[0][0]
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["is_first"] = (frame_id == 0)
        return dataset_dict

    def _get_metadata(self):
        meta = _get_builtin_metadata(self.task_key)
        image_root, json_file = _PREDEFINED_SPLITS_YOUTUBEVIS["youtubevis"][self.name]
        meta["image_root"] = osp.join(self.data_root, image_root) \
            if "://" not in image_root else image_root
        meta["json_file"] = osp.join(self.data_root, json_file) \
            if "://" not in image_root else osp.join(image_root, json_file)
        meta["evaluator_type"] = _PREDEFINED_SPLITS_YOUTUBEVIS["evaluator_type"]["youtubevis"]

        return meta

    def _apply_transforms(self, images, annotationss=None):
        """
        Apply a list of :class:`TransformGen` on the input image, and
        returns the transformed image and a list of transforms.

        We cannot simply create and return all transforms without
        applying it to the image, because a subsequent transform may
        need the output of the previous one.

        Args:
            transform_gens (list): list of :class:`TransformGen` instance to
                be applied.
            imgs (list[ndarray]): uint8 or floating point images with 1 or 3 channels.
            annotationss (list[list]): annotations
        Returns:
            ndarray: the transformed image
            TransformList: contain the transforms that's used.
        """

        if isinstance(self.transforms, dict):
            dataset_dict = {}
            for key, tfms in self.transforms.items():
                imgs = copy.deepcopy(images)
                annoss = copy.deepcopy(annotationss)
                for tfm in tfms:
                    imgs, annoss = tfm(imgs, annoss)
                dataset_dict[key] = (imgs, annoss)
            return dataset_dict, None
        else:
            for tfm in self.transforms:
                images, annotationss = tfm(images, annotationss)

            return images, annotationss

    def _load_annotations(self,
                          json_file,
                          image_root,
                          dataset_name=None):
        """
        Load a json file with COCO's instances annotation format.
        Args:
            json_file(str): full path to the json file in YouTube VIS video instance format.
            image_root(str): the directory where the videos in this json file exists.
            dataset_name(str): the name of the datasets(e.g., youtubevis)
        Returns:
             list[dict] a list of dicts in cvpods standard format
        """

        timer = Timer()
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self.visapi = YTVOS(json_file)
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f}".format(
                json_file, timer.seconds()))

        if dataset_name is not None:
            meta = self.meta
            cat_ids = sorted(self.visapi.getCatIds())  # [1, 40]
            cats = self.visapi.loadCats(cat_ids)  # list of dict
            thing_classes = [
                c["name"] for c in sorted(cats, key=lambda x: x["id"])
            ]
            meta["thing_classes"] = thing_classes

            id_map = {v: i for i, v in enumerate(cat_ids)}  # [0, 39]
            meta["thing_dataset_id_to_contiguous_id"] = id_map
        vid_ids = self.visapi.getVidIds()  # [1, 2238]
        vid_infos = []
        for i in vid_ids:
            info = self.visapi.loadVids([i])[0]
            vid_infos.append(info)
        img_ids = []
        for idx, vid_info in enumerate(vid_infos):
            for frame_id in range(len(vid_info['file_names'])):
                img_ids.append((idx, frame_id))  # idx:[0, 2237], frame:[0, len-1]

        self.vid_infos = vid_infos

        self.dataset_dicts = []  # record informations for whole datasets

        valid_img_ids = []
        self.vid2img2info = defaultdict(dict)

        for idx, frame_id in img_ids:  # loop for each image
            num_objs = 0
            record = {}  # record informations for one image

            record["file_name"] = osp.join(image_root, vid_infos[idx]["file_names"][frame_id])
            record["height"] = vid_infos[idx]["height"]
            record["width"] = vid_infos[idx]["width"]
            record["video_idx"] = idx  # 0-based
            record["frame_id"] = frame_id  # 0-based
            record["is_first"] = (frame_id == 0)
            video_id = record["video_id"] = vid_infos[idx]["id"]  # 1-based
            if self.train_mode:
                ann_ids = self.visapi.getAnnIds(vidIds=[video_id])
                ann_info = self.visapi.loadAnns(ann_ids)  # list of dict, all annotations of current image
                objs = []
                for ann in ann_info:
                    assert ann["video_id"] == video_id
                    obj = {}
                    bbox = ann['bboxes'][frame_id]  # xywh, list of float or None
                    if bbox is None:
                        continue
                    area = ann['areas'][frame_id]  # float or None
                    segm = ann['segmentations'][frame_id]  # dict(RLE) or None
                    x1, y1, w, h = bbox
                    if area <= 0 or w < 1 or h < 1:
                        continue
                    num_objs = num_objs + 1
                    obj["obj_id"] = ann["id"]  # 1-based instance id
                    obj["bbox"] = bbox
                    obj["iscrowd"] = ann["iscrowd"]
                    obj["category_id"] = ann["category_id"]  # 1-based

                    if segm:
                        if isinstance(segm, list):
                            segm = [
                                poly for poly in segm
                                if len(poly) % 2 == 0 and len(poly) >= 6
                            ]  # polygen
                        else:
                            assert sum(segm['counts']) == np.prod(segm['size'])  # RLE
                        obj["segmentation"] = segm

                    obj["bbox_mode"] = BoxMode.XYWH_ABS

                    if id_map:
                        obj["category_id"] = id_map[obj["category_id"]]  # 0-bases

                    objs.append(obj)
                if not num_objs:
                    continue
                record["annotations"] = objs

            valid_img_ids.append((idx, frame_id))  # 0-based
            self.vid2img2info[idx][frame_id] = record  # 0-based
            self.dataset_dicts.append(record)

        logger.info(
            "Trere are {} images in {} datasets and loaded {} videos and {} images in youtubevis datasets ".format(
                len(img_ids), self.name, len(vid_infos), len(valid_img_ids), json_file))

        return valid_img_ids

    def _check(self):
        if self.train_mode:
            for record in self.dataset_dicts:
                num_bbox = 0
                for obj in record['annotations']:
                    if obj['bbox'] is not None:
                        num_bbox += 1
                assert num_bbox


@DATASETS.register()
class YTVisDatasetLimit(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(YTVisDatasetLimit, self).__init__(cfg, dataset_name, transforms, is_train)

        self.task_key = 'youtubevislimit'
        self.meta = self._get_metadata()
        self.train_mode = is_train
        self.valid_img_ids = self._load_annotations(
            self.meta["json_file"],
            self.meta["image_root"],
            dataset_name)
        self._check()

        if is_train:
            self._set_group_flag()

        self.eval_with_gt = cfg.TEST.get("WITH_GT", False)

        self.data_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self.track_on = cfg.MODEL.TRACK_ON
        self.reference_range = cfg.DATASETS.REFERENCE_RANGE
        if cfg.DATASETS.ONLY_PREVIOUS is None:
            self.only_sample_prev = False
        else:
            self.only_sample_prev = cfg.DATASETS.ONLY_PREVIOUS

    def __len__(self):
        return len(self.valid_img_ids)

    def __getitem__(self, idx):
        if self.train_mode:
            return self._prepare_train_img(idx)
        else:
            return self._prepare_test_img(idx)

    def _sample_ref(self, idx):
        """
        Sample another frame in the same sequence as reference, if self.reference_range > 0, limit reference
        image sample range [-self.reference_range, self.reference_range], else do not limit sample range.
        """
        vid, frame_id = idx  # 0-based
        vid_info = self.vid_infos[vid]
        temp_range = self.reference_range
        while True:
            if temp_range < 0:
                sample_range = range(len(vid_info['file_names']))  # [0, len)
            else:
                start_index = max(0, frame_id - temp_range)
                end_index = min(frame_id + temp_range + 1, len(vid_info['file_names']))
                sample_range = range(start_index, end_index)
            valid_samples = []
            for i in sample_range:
                ref_idx = (vid, i)
                if i != frame_id and ref_idx in self.valid_img_ids:
                    valid_samples.append(ref_idx)
            if len(valid_samples) > 0:
                break
            else:
                logger.warning(f"In video:{vid} frame:{frame_id}, using reference range:{temp_range}. There are not "
                               f"any valid frame to sample reference image, so using temp_range{temp_range + 2} for "
                               f"next iteration.")
                temp_range += 2
        return random.choice(valid_samples)

    def _sample_ref_prev(self, idx):
        """
        Sample another frame in the same sequence as reference, if self.reference_range > 0, limit reference
        image sample range [-self.reference_range, self.reference_range], else do not limit sample range.
        """
        vid, frame_id = idx  # 0-based
        vid_info = self.vid_infos[vid]
        temp_range = self.reference_range
        invalid_frame_id = []
        end_index = frame_id
        while True:
            start_index = max(0, frame_id - temp_range)
            if start_index == end_index:
                logger.warning(f"In video:{vid} frame:{frame_id}, using reference range:{temp_range}. There are no "
                               f"valid previous frame can be sampled as reference frame, so random sample a new "
                               f"query frame in the same video")
                invalid_frame_id.append(frame_id)
                valid_samples_query = []
                sample_range = range(len(vid_info['file_names']))
                for i in sample_range:
                    ref_idx = (vid, i)
                    if i not in invalid_frame_id and ref_idx in self.valid_img_ids:
                        valid_samples_query.append(i)
                frame_id = random.choice(valid_samples_query)
                end_index = frame_id
                valid_samples_query.clear()
                continue

            sample_range = range(start_index, end_index)
            valid_samples = []
            for i in sample_range:
                ref_idx = (vid, i)
                if i != frame_id and ref_idx in self.valid_img_ids:
                    valid_samples.append(ref_idx)
            if len(valid_samples) > 0:
                break
            else:
                logger.warning(f"In video:{vid} frame:{frame_id}, using reference range:{temp_range}. There are not "
                               f"any valid frame to sample reference image, so using temp_range{temp_range + 2} for "
                               f"next iteration.")
                end_index = max(start_index, frame_id - temp_range)
                temp_range += 2

        return random.choice(valid_samples), frame_id

    def _prepare_train_img(self, idx):

        return_dict = {}
        query_vid, query_frame_id = self.valid_img_ids[idx]  # 0-based
        query_dataset_dict = copy.deepcopy(self.vid2img2info[query_vid][query_frame_id])  # 0-based
        if self.only_sample_prev:
            reference_id, query_frame_id = self._sample_ref_prev((query_vid, query_frame_id))
            reference_vid, reference_frame_id = reference_id
        else:
            reference_vid, reference_frame_id = self._sample_ref((query_vid, query_frame_id))  # 0-based
        reference_dataset_dict = copy.deepcopy(self.vid2img2info[reference_vid][reference_frame_id])

        query_img = read_image(query_dataset_dict["file_name"], format=self.data_format)
        check_image_size(query_dataset_dict, query_img)
        reference_img = read_image(reference_dataset_dict["file_name"], format=self.data_format)
        check_image_size(reference_dataset_dict, reference_img)

        if "annotations" in query_dataset_dict:
            query_annotations = query_dataset_dict.pop("annotations")  # list of dicts
        else:
            query_annotations = None
        if "annotations" in reference_dataset_dict:
            reference_annotations = reference_dataset_dict.pop("annotations")
        else:
            reference_annotations = None

        (query_img, reference_img), (query_annotations, reference_annotations) = \
            self._apply_transforms(images=[query_img, reference_img],
                                   annotationss=[query_annotations, reference_annotations])

        if query_annotations is not None:
            query_img_shape = query_img.shape[:2]
            query_instances = annotations_to_instances_vis(
                query_annotations, query_img_shape, mask_format=self.mask_format)
            query_dataset_dict["instances"] = filter_empty_instances(query_instances)

        if reference_annotations is not None:
            reference_img_shape = reference_img.shape[:2]
            reference_instances = annotations_to_instances_vis(
                reference_annotations, reference_img_shape, mask_format=self.mask_format)
            reference_dataset_dict["instances"] = filter_empty_instances(reference_instances)

        query_dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(query_img.transpose(2, 0, 1)))
        reference_dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(reference_img.transpose(2, 0, 1)))

        query_gt_ids = query_dataset_dict["instances"].instance_ids
        references_gt_ids = reference_dataset_dict["instances"].instance_ids

        gt_pids = [int((references_gt_ids == i).nonzero(as_tuple=False)) + 1 if i in references_gt_ids else 0 for i in
                   query_gt_ids]
        query_dataset_dict["instances"].gt_pids = torch.as_tensor(np.array(gt_pids))

        for k, v in query_dataset_dict.items():
            return_dict[k] = v
        for k, v in reference_dataset_dict.items():
            return_dict[k + "_reference"] = v
        del query_dataset_dict, reference_dataset_dict
        return return_dict

    def _prepare_test_img(self, idx):
        vid, frame_id = self.valid_img_ids[idx]  # 0-based
        dataset_dict = copy.deepcopy(self.vid2img2info[vid][frame_id])

        image = read_image(dataset_dict["file_name"], format=self.data_format)
        check_image_size(dataset_dict, image)
        if idx == len(self) - 1:
            is_last = True
        else:
            next_vid, next_frame_id = self.valid_img_ids[idx + 1]
            if next_vid != vid:
                is_last = True
            else:
                is_last = False
        dataset_dict["is_first"] = (frame_id == 0)
        dataset_dict["is_last"] = is_last
        if is_last:
            image = self._apply_transforms(images=[image], annotationss=[None])[0][0]
            dataset_dict["image"] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)))
        else:
            next_dataset_dict = copy.deepcopy(self.vid2img2info[next_vid][next_frame_id])
            next_image = read_image(next_dataset_dict["file_name"], format=self.data_format)
            check_image_size(next_dataset_dict, next_image)
            (image, next_image), _ = self._apply_transforms(images=[image, next_image], annotationss=[None, None])
            dataset_dict["image"] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)))
            dataset_dict["next_image"] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)))
        return dataset_dict

    def _get_metadata(self):
        meta = _get_builtin_metadata(self.task_key)
        image_root, json_file = _PREDEFINED_SPLITS_YOUTUBEVISLIMIT["youtubevislimit"][self.name]
        meta["image_root"] = osp.join(self.data_root, image_root) \
            if "://" not in image_root else image_root
        meta["json_file"] = osp.join(self.data_root, json_file) \
            if "://" not in image_root else osp.join(image_root, json_file)
        meta["evaluator_type"] = _PREDEFINED_SPLITS_YOUTUBEVISLIMIT["evaluator_type"]["youtubevis"]

        return meta

    def _apply_transforms(self, images, annotationss=None):
        """
        Apply a list of :class:`TransformGen` on the input image, and
        returns the transformed image and a list of transforms.

        We cannot simply create and return all transforms without
        applying it to the image, because a subsequent transform may
        need the output of the previous one.

        Args:
            transform_gens (list): list of :class:`TransformGen` instance to
                be applied.
            imgs (list[ndarray]): uint8 or floating point images with 1 or 3 channels.
            annotationss (list[list]): annotations
        Returns:
            ndarray: the transformed image
            TransformList: contain the transforms that's used.
        """

        if isinstance(self.transforms, dict):
            dataset_dict = {}
            for key, tfms in self.transforms.items():
                imgs = copy.deepcopy(images)
                annoss = copy.deepcopy(annotationss)
                for tfm in tfms:
                    imgs, annoss = tfm(imgs, annoss)
                dataset_dict[key] = (imgs, annoss)
            return dataset_dict, None
        else:
            for tfm in self.transforms:
                images, annotationss = tfm(images, annotationss)

            return images, annotationss

    def _load_annotations(self,
                          json_file,
                          image_root,
                          dataset_name=None):
        """
        Load a json file with COCO's instances annotation format.
        Args:
            json_file(str): full path to the json file in YouTube VIS video instance format.
            image_root(str): the directory where the videos in this json file exists.
            dataset_name(str): the name of the datasets(e.g., youtubevis)
        Returns:
             list[dict] a list of dicts in cvpods standard format
        """

        timer = Timer()
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self.visapi = YTVOS(json_file)
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f}".format(
                json_file, timer.seconds()))

        if dataset_name is not None:
            meta = self.meta
            cat_ids = sorted(self.visapi.getCatIds())  # [1, 40]
            cats = self.visapi.loadCats(cat_ids)  # list of dict
            thing_classes = [
                c["name"] for c in sorted(cats, key=lambda x: x["id"])
            ]
            meta["thing_classes"] = thing_classes

            id_map = {v: i for i, v in enumerate(cat_ids)}  # [0, 39]
            meta["thing_dataset_id_to_contiguous_id"] = id_map
        vid_ids = self.visapi.getVidIds()  # [1, 2238]
        vid_infos = []
        for i in vid_ids:
            info = self.visapi.loadVids([i])[0]
            vid_infos.append(info)
        img_ids = []
        for idx, vid_info in enumerate(vid_infos):  # Loop for each video
            for frame_id in range(len(vid_info['file_names'])):  # Loop for each frame
                img_ids.append((idx, frame_id))  # idx:[0, 2237], frame:[0, len-1]

        self.vid_infos = vid_infos

        self.dataset_dicts = []  # record informations for whole datasets

        valid_img_ids = []
        self.vid2img2info = defaultdict(dict)

        for idx, frame_id in img_ids:  # loop for each image
            num_objs = 0
            record = {}  # record informations for one image

            record["file_name"] = osp.join(image_root, vid_infos[idx]["file_names"][frame_id])
            record["height"] = vid_infos[idx]["height"]
            record["width"] = vid_infos[idx]["width"]
            record["video_idx"] = idx  # 0-based
            record["frame_id"] = frame_id  # 0-based
            record["is_first"] = (frame_id == 0)
            video_id = record["video_id"] = vid_infos[idx]["id"]  # 1-based
            if self.train_mode:
                ann_ids = self.visapi.getAnnIds(vidIds=[video_id])
                ann_info = self.visapi.loadAnns(ann_ids)  # list of dict, all annotations of current image
                objs = []
                for ann in ann_info:
                    assert ann["video_id"] == video_id
                    obj = {}
                    bbox = ann['bboxes'][frame_id]  # xywh, list of float or None
                    if bbox is None:
                        continue
                    area = ann['areas'][frame_id]  # float or None
                    segm = ann['segmentations'][frame_id]  # dict(RLE) or None
                    x1, y1, w, h = bbox
                    if area <= 0 or w < 1 or h < 1:
                        continue
                    num_objs = num_objs + 1
                    obj["obj_id"] = ann["id"]  # 1-based instance id
                    obj["bbox"] = bbox
                    obj["iscrowd"] = ann["iscrowd"]
                    obj["category_id"] = ann["category_id"]  # 1-based

                    if segm:
                        if isinstance(segm, list):
                            segm = [
                                poly for poly in segm
                                if len(poly) % 2 == 0 and len(poly) >= 6
                            ]  # polygen
                        else:
                            assert sum(segm['counts']) == np.prod(segm['size'])  # RLE
                        obj["segmentation"] = segm

                    obj["bbox_mode"] = BoxMode.XYWH_ABS

                    if id_map:
                        obj["category_id"] = id_map[obj["category_id"]]  # 0-bases

                    objs.append(obj)
                if not num_objs:
                    continue
                record["annotations"] = objs

            valid_img_ids.append((idx, frame_id))  # 0-based
            self.vid2img2info[idx][frame_id] = record  # 0-based
            self.dataset_dicts.append(record)

        logger.info(
            "Trere are {} images in {} datasets and loaded {} videos and {} images in youtubevis datasets ".format(
                len(img_ids), self.name, len(vid_infos), len(valid_img_ids), json_file))

        return valid_img_ids

    def _check(self):
        if self.train_mode:
            for record in self.dataset_dicts:
                num_bbox = 0
                for obj in record['annotations']:
                    if obj['bbox'] is not None:
                        num_bbox += 1
                assert num_bbox


@DATASETS.register()
class YTVisDataset2021Limit(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(YTVisDataset2021Limit, self).__init__(cfg, dataset_name, transforms, is_train)

        self.task_key = 'youtubevis2021limit'
        self.meta = self._get_metadata()
        self.train_mode = is_train
        self.valid_img_ids = self._load_annotations(
            self.meta["json_file"],
            self.meta["image_root"],
            dataset_name)
        self._check()

        if is_train:
            self._set_group_flag()

        self.eval_with_gt = cfg.TEST.get("WITH_GT", False)

        self.data_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self.track_on = cfg.MODEL.TRACK_ON
        self.reference_range = cfg.DATASETS.REFERENCE_RANGE
        self.only_sample_prev = cfg.DATASETS.ONLY_PREVIOUS

    def __len__(self):
        return len(self.valid_img_ids)

    def __getitem__(self, idx):
        if self.train_mode:
            return self._prepare_train_img(idx)
        else:
            return self._prepare_test_img(idx)

    def _sample_ref(self, idx):
        """
        Sample another frame in the same sequence as reference, if self.reference_range > 0, limit reference
        image sample range [-self.reference_range, self.reference_range], else do not limit sample range.
        """
        vid, frame_id = idx  # 0-based
        vid_info = self.vid_infos[vid]
        temp_range = self.reference_range
        while True:
            if temp_range < 0:
                sample_range = range(len(vid_info['file_names']))  # [0, len)
            else:
                start_index = max(0, frame_id - temp_range)
                end_index = min(frame_id + temp_range + 1, len(vid_info['file_names']))
                sample_range = range(start_index, end_index)
            valid_samples = []
            for i in sample_range:
                ref_idx = (vid, i)
                if i != frame_id and ref_idx in self.valid_img_ids:
                    valid_samples.append(ref_idx)
            if len(valid_samples) > 0:
                break
            else:
                logger.warning(f"In video:{vid} frame:{frame_id}, using reference range:{temp_range}. There are not "
                               f"any valid frame to sample reference image, so using temp_range{temp_range + 2} for "
                               f"next iteration.")
                temp_range += 2
        return random.choice(valid_samples)

    def _sample_ref_prev(self, idx):
        """
        Sample another frame in the same sequence as reference, if self.reference_range > 0, limit reference
        image sample range [-self.reference_range, self.reference_range], else do not limit sample range.
        """
        vid, frame_id = idx  # 0-based
        vid_info = self.vid_infos[vid]
        temp_range = self.reference_range
        invalid_frame_id = []
        end_index = frame_id
        while True:
            start_index = max(0, frame_id - temp_range)
            if start_index == end_index:
                logger.warning(f"In video:{vid} frame:{frame_id}, using reference range:{temp_range}. There are no "
                               f"valid previous frame can be sampled as reference frame, so random sample a new "
                               f"query frame in the same video")
                invalid_frame_id.append(frame_id)
                valid_samples_query = []
                sample_range = range(len(vid_info['file_names']))
                for i in sample_range:
                    ref_idx = (vid, i)
                    if i not in invalid_frame_id and ref_idx in self.valid_img_ids:
                        valid_samples_query.append(i)
                frame_id = random.choice(valid_samples_query)
                end_index = frame_id
                valid_samples_query.clear()
                continue

            sample_range = range(start_index, end_index)
            valid_samples = []
            for i in sample_range:
                ref_idx = (vid, i)
                if i != frame_id and ref_idx in self.valid_img_ids:
                    valid_samples.append(ref_idx)
            if len(valid_samples) > 0:
                break
            else:
                logger.warning(f"In video:{vid} frame:{frame_id}, using reference range:{temp_range}. There are not "
                               f"any valid frame to sample reference image, so using temp_range{temp_range + 2} for "
                               f"next iteration.")
                end_index = max(start_index, frame_id - temp_range)
                temp_range += 2

        return random.choice(valid_samples), frame_id

    def _prepare_train_img(self, idx):

        return_dict = {}
        query_vid, query_frame_id = self.valid_img_ids[idx]  # 0-based
        query_dataset_dict = copy.deepcopy(self.vid2img2info[query_vid][query_frame_id])  # 0-based
        if self.only_sample_prev:
            reference_id, query_frame_id = self._sample_ref_prev((query_vid, query_frame_id))
            reference_vid, reference_frame_id = reference_id
        else:
            reference_vid, reference_frame_id = self._sample_ref((query_vid, query_frame_id))  # 0-based
        reference_dataset_dict = copy.deepcopy(self.vid2img2info[reference_vid][reference_frame_id])

        query_img = read_image(query_dataset_dict["file_name"], format=self.data_format)
        check_image_size(query_dataset_dict, query_img)
        reference_img = read_image(reference_dataset_dict["file_name"], format=self.data_format)
        check_image_size(reference_dataset_dict, reference_img)

        if "annotations" in query_dataset_dict:
            query_annotations = query_dataset_dict.pop("annotations")  # list of dicts
        else:
            query_annotations = None
        if "annotations" in reference_dataset_dict:
            reference_annotations = reference_dataset_dict.pop("annotations")
        else:
            reference_annotations = None

        (query_img, reference_img), (query_annotations, reference_annotations) = \
            self._apply_transforms(images=[query_img, reference_img],
                                   annotationss=[query_annotations, reference_annotations])

        if query_annotations is not None:
            query_img_shape = query_img.shape[:2]
            query_instances = annotations_to_instances_vis(
                query_annotations, query_img_shape, mask_format=self.mask_format)
            query_dataset_dict["instances"] = filter_empty_instances(query_instances)

        if reference_annotations is not None:
            reference_img_shape = reference_img.shape[:2]
            reference_instances = annotations_to_instances_vis(
                reference_annotations, reference_img_shape, mask_format=self.mask_format)
            reference_dataset_dict["instances"] = filter_empty_instances(reference_instances)

        query_dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(query_img.transpose(2, 0, 1)))
        reference_dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(reference_img.transpose(2, 0, 1)))

        query_gt_ids = query_dataset_dict["instances"].instance_ids
        references_gt_ids = reference_dataset_dict["instances"].instance_ids

        gt_pids = [int((references_gt_ids == i).nonzero(as_tuple=False)) + 1 if i in references_gt_ids else 0 for i in
                   query_gt_ids]
        query_dataset_dict["instances"].gt_pids = torch.as_tensor(np.array(gt_pids))

        for k, v in query_dataset_dict.items():
            return_dict[k] = v
        for k, v in reference_dataset_dict.items():
            return_dict[k + "_reference"] = v
        del query_dataset_dict, reference_dataset_dict
        return return_dict

    def _prepare_test_img(self, idx):
        vid, frame_id = self.valid_img_ids[idx]  # 0-based
        dataset_dict = copy.deepcopy(self.vid2img2info[vid][frame_id])

        image = read_image(dataset_dict["file_name"], format=self.data_format)
        check_image_size(dataset_dict, image)
        if idx == len(self) - 1:
            is_last = True
        else:
            next_vid, next_frame_id = self.valid_img_ids[idx + 1]
            if next_vid != vid:
                is_last = True
            else:
                is_last = False
        dataset_dict["is_first"] = (frame_id == 0)
        dataset_dict["is_last"] = is_last
        if is_last:
            image = self._apply_transforms(images=[image], annotationss=[None])[0][0]
            dataset_dict["image"] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)))
        else:
            next_dataset_dict = copy.deepcopy(self.vid2img2info[next_vid][next_frame_id])
            next_image = read_image(next_dataset_dict["file_name"], format=self.data_format)
            check_image_size(next_dataset_dict, next_image)
            (image, next_image), _ = self._apply_transforms(images=[image, next_image], annotationss=[None, None])
            dataset_dict["image"] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)))
            dataset_dict["next_image"] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)))
        return dataset_dict

    def _get_metadata(self):
        meta = _get_builtin_metadata(self.task_key)
        image_root, json_file = _PREDEFINED_SPLITS_YOUTUBEVIS2021LIMIT["youtubevis2021limit"][self.name]
        meta["image_root"] = osp.join(self.data_root, image_root) \
            if "://" not in image_root else image_root
        meta["json_file"] = osp.join(self.data_root, json_file) \
            if "://" not in image_root else osp.join(image_root, json_file)
        meta["evaluator_type"] = _PREDEFINED_SPLITS_YOUTUBEVIS2021LIMIT["evaluator_type"]["youtubevis"]

        return meta

    def _apply_transforms(self, images, annotationss=None):
        """
        Apply a list of :class:`TransformGen` on the input image, and
        returns the transformed image and a list of transforms.

        We cannot simply create and return all transforms without
        applying it to the image, because a subsequent transform may
        need the output of the previous one.

        Args:
            transform_gens (list): list of :class:`TransformGen` instance to
                be applied.
            imgs (list[ndarray]): uint8 or floating point images with 1 or 3 channels.
            annotationss (list[list]): annotations
        Returns:
            ndarray: the transformed image
            TransformList: contain the transforms that's used.
        """

        if isinstance(self.transforms, dict):
            dataset_dict = {}
            for key, tfms in self.transforms.items():
                imgs = copy.deepcopy(images)
                annoss = copy.deepcopy(annotationss)
                for tfm in tfms:
                    imgs, annoss = tfm(imgs, annoss)
                dataset_dict[key] = (imgs, annoss)
            return dataset_dict, None
        else:
            for tfm in self.transforms:
                images, annotationss = tfm(images, annotationss)

            return images, annotationss

    def _load_annotations(self,
                          json_file,
                          image_root,
                          dataset_name=None):
        """
        Load a json file with COCO's instances annotation format.
        Args:
            json_file(str): full path to the json file in YouTube VIS video instance format.
            image_root(str): the directory where the videos in this json file exists.
            dataset_name(str): the name of the datasets(e.g., youtubevis)
        Returns:
             list[dict] a list of dicts in cvpods standard format
        """

        timer = Timer()
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self.visapi = YTVOS(json_file)
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f}".format(
                json_file, timer.seconds()))

        if dataset_name is not None:
            meta = self.meta
            cat_ids = sorted(self.visapi.getCatIds())  # [1, 40]
            cats = self.visapi.loadCats(cat_ids)  # list of dict
            thing_classes = [
                c["name"] for c in sorted(cats, key=lambda x: x["id"])
            ]
            meta["thing_classes"] = thing_classes

            id_map = {v: i for i, v in enumerate(cat_ids)}  # [0, 39]
            meta["thing_dataset_id_to_contiguous_id"] = id_map
        vid_ids = self.visapi.getVidIds()  # [1, 2238]
        vid_infos = []
        for i in vid_ids:
            info = self.visapi.loadVids([i])[0]
            vid_infos.append(info)
        img_ids = []
        for idx, vid_info in enumerate(vid_infos):  # Loop for each video
            for frame_id in range(len(vid_info['file_names'])):  # Loop for each frame
                img_ids.append((idx, frame_id))  # idx:[0, 2237], frame:[0, len-1]

        self.vid_infos = vid_infos

        self.dataset_dicts = []  # record informations for whole datasets

        valid_img_ids = []
        self.vid2img2info = defaultdict(dict)

        for idx, frame_id in img_ids:  # loop for each image
            num_objs = 0
            record = {}  # record informations for one image

            record["file_name"] = osp.join(image_root, vid_infos[idx]["file_names"][frame_id])
            record["height"] = vid_infos[idx]["height"]
            record["width"] = vid_infos[idx]["width"]
            record["video_idx"] = idx  # 0-based
            record["frame_id"] = frame_id  # 0-based
            record["is_first"] = (frame_id == 0)
            video_id = record["video_id"] = vid_infos[idx]["id"]  # 1-based
            if self.train_mode:
                ann_ids = self.visapi.getAnnIds(vidIds=[video_id])
                ann_info = self.visapi.loadAnns(ann_ids)  # list of dict, all annotations of current image
                objs = []
                for ann in ann_info:
                    assert ann["video_id"] == video_id
                    obj = {}
                    bbox = ann['bboxes'][frame_id]  # xywh, list of float or None
                    if bbox is None:
                        continue
                    area = ann['areas'][frame_id]  # float or None
                    segm = ann['segmentations'][frame_id]  # dict(RLE) or None
                    x1, y1, w, h = bbox
                    if area <= 0 or w < 1 or h < 1:
                        continue
                    num_objs = num_objs + 1
                    obj["obj_id"] = ann["id"]  # 1-based instance id
                    obj["bbox"] = bbox
                    obj["iscrowd"] = ann["iscrowd"]
                    obj["category_id"] = ann["category_id"]  # 1-based

                    if segm:
                        if isinstance(segm, list):
                            segm = [
                                poly for poly in segm
                                if len(poly) % 2 == 0 and len(poly) >= 6
                            ]  # polygen
                        else:
                            assert sum(segm['counts']) == np.prod(segm['size'])  # RLE
                        obj["segmentation"] = segm

                    obj["bbox_mode"] = BoxMode.XYWH_ABS

                    if id_map:
                        obj["category_id"] = id_map[obj["category_id"]]  # 0-bases

                    objs.append(obj)
                if not num_objs:
                    continue
                record["annotations"] = objs

            valid_img_ids.append((idx, frame_id))  # 0-based
            self.vid2img2info[idx][frame_id] = record  # 0-based
            self.dataset_dicts.append(record)

        logger.info(
            "Trere are {} images in {} datasets and loaded {} videos and {} images in youtubevis datasets ".format(
                len(img_ids), self.name, len(vid_infos), len(valid_img_ids), json_file))

        return valid_img_ids

    def _check(self):
        if self.train_mode:
            for record in self.dataset_dicts:
                num_bbox = 0
                for obj in record['annotations']:
                    if obj['bbox'] is not None:
                        num_bbox += 1
                assert num_bbox
