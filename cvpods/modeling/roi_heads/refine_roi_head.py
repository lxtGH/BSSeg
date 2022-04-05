# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict, List, Union
from functools import partial
import numpy as np
import torch

from cvpods.layers import cat
from cvpods.structures import Instances, BitMasks
from .refine_mask_head import generate_block_target

from cvpods.modeling.roi_heads import StandardROIHeads, select_foreground_proposals


logger = logging.getLogger(__name__)


def refinemask_inference(stage_instance_preds, instances):
    """
    Convert stage_instance_preds to the final instance predictions. For each
    predicted box, the according predicted mask is attached to the instance
    by adding a new "pred_masks" field to pred_instances.

    Args:
        stage_instance_preds (list of Tensor): instance predictions for the initial prediction
            and three refinement predictions. Here we only use the latter three refinement instance predictions.
            shape_i=(num_instances, 1, 14/28/56/112, 14, 28, 56, 112)
        instances (list of Instances): A list of N Instances, where N is the numberof images
            in the batch.

    Returns:
         None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for each predicted instance. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    stage_instance_refinement_preds = stage_instance_preds[1:]  # 0 is the initial instance prediction
    for idx in range(len(stage_instance_refinement_preds) - 1):
        instance_pred = stage_instance_refinement_preds[idx].squeeze(1).sigmoid() >= 0.5
        # (num_instances, 1, 28/56, 28/56)
        non_boundary_mask = (generate_block_target(instance_pred, boundary_width=1) != 1).unsqueeze(1)
        # (num_instances, 1, 28/56, 28/56)
        non_boundary_mask = torch.nn.functional.interpolate(
            non_boundary_mask.float(),
            stage_instance_refinement_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True) >= 0.5
        # Upsamples non-boundary region (num_instances, 1, 56/112, 56/112)
        pre_pred = torch.nn.functional.interpolate(
            stage_instance_refinement_preds[idx],
            stage_instance_refinement_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True)
        # Upsamples this stage's instance prediction (num_instances, 1, 56/112, 56/112)
        stage_instance_refinement_preds[idx + 1][non_boundary_mask] = pre_pred[non_boundary_mask]
        # boundary-aware refinement
    instance_pred = stage_instance_refinement_preds[-1]
    # The final instance prediction (num_instances, 1, 112, 112)
    num_instances_per_image = [len(instance) for instance in instances]
    instance_pred = instance_pred.split(num_instances_per_image, dim=0)
    # list of tensor, instance prediction for each image, shape_i = [num_instances_per_image, 1, 112, 112]
    for prob, instance in zip(instance_pred, instances):
        instance.pred_masks = prob
    return instances


def get_refinemask_target(instances, stage_sup_size):
    """
    Get the targets for training the refine mask

    Args:
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        stage_sup_size (list[Int]): height/width of each stage's instance prediction
    Returns:
        semantic_targets (list[Tensor]): A list of N instance targets, where N is the number of
        stage, i.e. len(stage_sup_size) shape = [num_instances, 14/28/56/112, 14/28/56/112]
        semantic_targets (list[Tensor]): A list of B semantic targets, where B is the batch size, shape = [imh_i, imw_i]
    """

    def _generate_instance_targets(pos_proposals, gt_masks, mask_size=None):
        device = pos_proposals.device
        proposals_np = pos_proposals.cpu().numpy()  # [num_pos_boxes, 4]
        maxh, maxw = gt_masks.image_size
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)

        # crop and resize the instance mask
        instance_targets = gt_masks.crop_and_resize(
            proposals_np, mask_size, device=device, ).float().to(device)
        # instance_targets = torch.from_numpy(resize_masks).float().to(device)  # Tensor(Bitmaps)
        return instance_targets

    stage_instance_targets = []
    semantic_targets = []
    for instances_per_image in instances:           # Loop for each images
        if len(instances_per_image) == 0:
            continue
        stage_instance_targets_pre_image = [_generate_instance_targets(
            instances_per_image.proposal_boxes.tensor, instances_per_image.gt_masks, mask_size=mask_size)
            for mask_size in stage_sup_size]
        stage_instance_targets.append(stage_instance_targets_pre_image)

        assert isinstance(instances_per_image.gt_masks, BitMasks)
        instance_masks = instances_per_image.gt_masks.tensor
        semantic_targets.append(instance_masks.max(dim=0, keep_dim=True))
    pcat = partial(cat, dim=0)
    stage_instance_targets = list(map(pcat, zip(*stage_instance_targets)))

    semantic_targets = semantic_targets

    return stage_instance_targets, semantic_targets


class RefineRoIHead(StandardROIHeads):
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
        super(RefineRoIHead, self).__init__(cfg, input_shape)


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
            proposal_boxes_tensor = [proposal_box.tensor for proposal_box in proposal_boxes]
            proposal_labels = [x.gt_classes for x in proposals]
            instance_features = self.mask_pooler(features, proposal_boxes)          # [num_boxes, c, h, w]
            stage_instance_preds, semantic_pred = self.mask_head(instance_features, features[0],
                                                                 proposal_boxes_tensor, proposal_labels)
            stage_instance_targets, semantic_targets = get_refinemask_target(instances, self.mask_head.stage_sup_size)
            for i in range(len(semantic_targets)):
                semantic_targets[i] = torch.nn.functional.interpolate(
                    semantic_targets[i].unsqueeze(1), semantic_pred.shape[-2],
                    mode='bilinear', align_corners=False).squeeze(1)
            semantic_targets = cat(semantic_targets, dim=0)                         # [bs, h, w]    stride4

            loss_instance, loss_semantic = self.mask_head.loss(stage_instance_preds, semantic_pred,
                                                               stage_instance_targets, semantic_targets)
            return {"loss_mask_instance": loss_instance, "loss_semantic_mask": loss_semantic}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            pred_boxes_tensor = [pred_box.tensor for pred_box in pred_boxes]
            pred_labels = [x.pred_classes for x in instances]
            instance_features = self.mask_pooler(features, pred_boxes)
            stage_instance_preds, _ = self.mask_head(instance_features, features[0],
                                                                 pred_boxes_tensor, pred_labels)
            refinemask_inference(stage_instance_preds, instances)
            return instances
