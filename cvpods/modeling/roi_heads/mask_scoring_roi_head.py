# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict, List, Union
import torch

from cvpods.layers import ShapeSpec, cat
from cvpods.structures import Instances
from .mask_head import mask_rcnn_inference, mask_rcnn_loss

from cvpods.modeling.roi_heads import StandardROIHeads, select_foreground_proposals


logger = logging.getLogger(__name__)


class MaskScoringRoIHead(StandardROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(MaskScoringRoIHead, self).__init__(cfg, input_shape)
        self._init_mask_iou_head(cfg)

    def _init_mask_iou_head(self, cfg):
        #fmt: off
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_iou_head = cfg.build_mask_iou_head(cfg,
                                                     ShapeSpec(
                                                         channels=in_channels,
                                                         width=pooler_resolution,
                                                         height=pooler_resolution))

    def _forward_mask(
        self, features: List[torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            mask_logits = self.mask_head(mask_features)
            loss_mask, mask_target, pred_mask_logits = mask_rcnn_loss(mask_logits, proposals, return_for_mask_scoring=True)
            mask_iou_target = self.mask_iou_head.get_target(mask_logits, proposals, mask_target)
            mask_iou_pred = self.mask_iou_head(mask_features, pred_mask_logits)
            loss_mask_iou = self.mask_iou_head.loss(mask_iou_pred, mask_iou_target, proposals)
            return {"loss_mask": loss_mask, "loss_mask_iou": loss_mask_iou}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)          # num_boxes, 256, 14, 14
            mask_logits = self.mask_head(mask_features)                     # num_boxes, C, 28, 28
            class_pred = mask_rcnn_inference(mask_logits, instances)
            mask_iou_pred = self.mask_iou_head(mask_features, cat(instances.pred_masks))
            indices = torch.arange(mask_iou_pred.shape[0])
            mask_iou_pred = mask_iou_pred[indices, class_pred[:, None]]
            num_boxes_per_image = [len(i) for i in instances]
            mask_iou_pred = mask_iou_pred.split(num_boxes_per_image, dim=0)
            for instances_per_image, iou_score in zip(instances, mask_iou_pred):
                instances_per_image.objectness_logits *= iou_score
            return instances

