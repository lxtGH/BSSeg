#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   coco_caption.py
@Author             :   Xiangtai Li
'''


import os.path as osp
import numpy as np
import torch

import torchvision

import cvpods
from ..registry import DATASETS
from .paths_route import _PREDEFINED_SPLITS_COCOCAPTIONS


@DATASETS.register()
class COCOCaptionsDataset(torchvision.datasets.coco.CocoCaptions):
    def __init__(
        self, cfg, dataset_name,
        transforms=None, is_train=True
    ):
        self.data_root = osp.join(
            osp.split(osp.split(cvpods.__file__)[0])[0], "datasets")
        root, ann_file = self._get_root(dataset_name)
        self.root = osp.join(self.data_root, root)
        self.ann_file = osp.join(self.data_root, ann_file)
        self.meta = {}
        super(COCOCaptionsDataset, self).__init__(self.root, self.ann_file)

        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        # hack
        remove_images_without_annotations = True
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anno = self.coco.loadAnns(ann_ids)
                if len(anno) > 0:
                    ids.append(img_id)
            self.ids = ids

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

        # This flag is specially designed for COCOCaptions dataset.
        # When set to True, instead of passing the captions as target,
        # we convert captions to multi-label binary classification targets,
        # which can be used to train weakly supervised object detectors.
        # We do not use it (hack)
        self.multilabel_mode = False
        if is_train:
            self._set_group_flag()

    def __getitem__(self, idx):
        dataset_dict ={}
        img, anno = super(COCOCaptionsDataset, self).__getitem__(idx)
        if self.multilabel_mode:
            anno = self.convert_to_multilabel_anno(anno)
        else:
            # anno is a list of sentences. Pick one randomly.
            # TODO use a more deterministic approach, especially for validation
            anno = np.random.choice(anno)

        # if self._transforms is not None:
        #     img, _ = self._transforms(img, None)

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(img)).permute(2, 0, 1)
        #.permute(1, 2, 0)  # (h,w,3 -> 3,h,w)
        # print(dataset_dict["image"].shape)

        dataset_dict["annotation"] = anno

        return dataset_dict

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def convert_to_multilabel_anno(self, sentence_list):
        anno = np.zeros((self.num_categories), dtype=np.float32)
        for cid, cind in self.json_category_id_to_contiguous_id.items():
            cname = self.categories[cid].lower()
            for sent in sentence_list:
                if cname in sent.lower():
                    anno[cind] = 1
        return anno

    def set_class_labels(self, categories, json_category_id_to_contiguous_id):
        '''
        For multi-label mode only
        Should be called to register the list of categories before calling __getitem__()
        '''
        self.categories = categories
        self.json_category_id_to_contiguous_id = json_category_id_to_contiguous_id
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.num_categories = max(list(self.contiguous_category_id_to_json_id.keys())) + 1

    def _get_root(self, dataset_name):
        root, ann_file = _PREDEFINED_SPLITS_COCOCAPTIONS["cococaption"][dataset_name]
        return root, ann_file

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.aspect_ratios = np.zeros(len(self), dtype=np.uint8)