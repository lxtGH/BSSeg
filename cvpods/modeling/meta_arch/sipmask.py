# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

import math
import numpy as np

from cvpods.structures import ImageList, pairwise_iou_tensor
from cvpods.structures.instances import Instances
from cvpods.structures.boxes import Boxes, align_box_iou_tensor
from cvpods.structures.masks import polygons_to_bitmask
from cvpods.layers import ShapeSpec, cat, batched_nms, NaiveSyncBatchNorm, NaiveGroupNorm
from cvpods.layers.deform_conv import DeformConv, DFConv2d
from cvpods.layers.crop_split import CropSplit
from cvpods.layers.crop_split_gt import CropSplitGT
from cvpods.layers.misc import aligned_bilinear
from cvpods.modeling.losses import IOULoss, sigmoid_focal_loss_jit
from cvpods.utils.distributed.comm import get_world_size, reduce_sum
from cvpods.layers.conv_with_kaiming_uniform import conv_with_kaiming_uniform


INF = 100000000


def ml_nms(boxlist, nms_thresh, max_proposals=-1):
    if nms_thresh < 0:
        return boxlist
    boxes = boxlist.pred_boxes.tensor
    scores = boxlist.scores
    labels = boxlist.pred_classes
    keep = batched_nms(boxes, scores, labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[:max_proposals]
    boxlist = boxlist[keep]
    return boxlist


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(( (boxes[:, 2:] + boxes[:, :2])/2,     # cx, cy
                        boxes[:, 2:] - boxes[:, :2]  ), 1)  # w, h


def accuracy(pred, target, topk=1):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


class SipMaskTrackHead(nn.Module):
    def __init__(self, cfg):
        super(SipMaskTrackHead, self).__init__()
        self.stacked_convs = cfg.MODEL.SIPMASK.TRACKHEAD.NUM_TRACK_CONVS
        self.use_deformable = cfg.MODEL.SIPMASK.TRACKHEAD.USE_DEFORMABLE
        self.in_channels = cfg.MODEL.SIPMASK.TRACKHEAD.IN_CHANNELS
        self.feat_channels = cfg.MODEL.SIPMASK.TRACKHEAD.FEAT_CHANNELS
        self.norm = None if cfg.MODEL.SIPMASK.TRACKHEAD.NORM == 'none' else cfg.MODEL.SIPMASK.TRACKHEAD.NORM
        self.in_features = cfg.MODEL.SIPMASK.TRACKHEAD.IN_FEATURES
        self._init_layers()

    def _init_layers(self):
        tower = []
        for i in range(self.stacked_convs):
            in_channels = self.in_channels if i == 0 else self.feat_channels
            if self.use_deformable and i == self.stacked_convs - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d
            tower.append(conv_func(in_channels=in_channels,
                                   out_channels=self.feat_channels,
                                   kernel_size=3,
                                   padding=1,
                                   stride=1,
                                   bias=self.norm is None))
            if self.norm == "GN":
                tower.append(nn.GroupNorm(32, in_channels))
            elif self.norm == "NaiveGN":
                tower.append(NaiveGroupNorm(32, in_channels))
            elif self.norm == "BN":
                tower.append(ModuleListDial([
                    nn.BatchNorm2d(in_channels) for _ in range(self.num_levels)
                ]))
            elif self.norm == "SyncBN":
                tower.append(ModuleListDial([
                    NaiveSyncBatchNorm(in_channels) for _ in range(self.num_levels)
                ]))
            tower.append(nn.ReLU())
        self.add_module('track_tower', nn.Sequential(*tower))

        self.sipmask_track = nn.Conv2d(self.feat_channels * 3, 512, 1, padding=0)

    def _train_forward(self, query_feats, reference_feats):
        count = 0
        query_track_feats = []
        reference_track_feats = []
        for query_feat, reference_feat in zip(query_feats, reference_feats):
            if count < 3:
                query_track_feat = self.track_tower(query_feat)
                query_track_feat = F.interpolate(query_track_feat, scale_factor=(2 ** count),
                                                 mode='bilinear', align_corners=False)
                query_track_feats.append(query_track_feat)
                reference_track_feat = self.track_tower(reference_feat)
                reference_track_feat = F.interpolate(reference_track_feat, scale_factor=(2 ** count),
                                                     mode='bilinear', align_corners=False)
                reference_track_feats.append(reference_track_feat)
            else:
                break
            count += 1
        query_track_feats = cat(query_track_feats, dim=1)
        query_track = self.sipmask_track(query_track_feats)
        reference_track_feats = cat(reference_track_feats, dim=1)
        reference_track = self.sipmask_track(reference_track_feats)
        return query_track, reference_track

    def _inference_forward(self, query_feats):
        count = 0
        query_track_feats = []
        for query_feat in query_feats:
            if count < 3:
                query_track_feat = self.track_tower(query_feat)
                query_track_feat = F.interpolate(query_track_feat, scale_factor=(2 ** count),
                                                 mode='bilinear', align_corners=False)
                query_track_feats.append(query_track_feat)
            else:
                break
            count += 1
        query_track_feats = cat(query_track_feats, dim=1)
        query_track = self.sipmask_track(query_track_feats)
        return query_track

    def forward(self, query_feats, reference_feats=None):
        query_feats = [query_feats[f] for f in self.in_features]
        if self.training:
            reference_feats = [reference_feats[f] for f in self.in_features]
            query_track, reference_track = self._train_forward(query_feats, reference_feats)
            return query_track, reference_track
        else:
            query_track = self._inference_forward(query_feats)
            return query_track


class FeatureAlign(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAlign, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(4,
                                     deformable_groups * offset_channels,
                                     1,
                                     bias=False)
        self.conv_adaption = DeformConv(in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        padding=(kernel_size-1) // 2,
                                        deformable_groups=deformable_groups,
                                        bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.GroupNorm(32, in_channels)

    def init_weights(self):

        nn.init.normal_(self.conv_offset.weight, std=0.01)
        nn.init.normal_(self.conv_adaption.weight, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.norm(self.conv_adaption(x, offset)))
        return x


class CoefficientPredictor(nn.Module):
    def __init__(self, in_channels, num_basic_mask, num_sub_regions):
        super(CoefficientPredictor, self).__init__()
        self.sip_cof = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_basic_mask*num_sub_regions,
                                 kernel_size=3,
                                 padding=1)

    def forward(self, cls_towers):
        cof_preds = []
        for cls_tower in cls_towers:
            cof_pred = self.sip_cof(cls_tower)
            cof_preds.append(cof_pred)
        return cof_preds


class BasicMaskGenerator(nn.Module):
    def __init__(self, in_channels, channels, num_basic_mask):
        super(BasicMaskGenerator, self).__init__()
        self.basic_mask_lat0 = nn.Conv2d(in_channels=in_channels, out_channels=channels,
                                         kernel_size=3, padding=1)
        self.basic_mask_lat = nn.Conv2d(in_channels=channels, out_channels=num_basic_mask, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        basic_masks_inter = self.relu(self.basic_mask_lat0(x))
        basic_masks = self.relu(self.basic_mask_lat(basic_masks_inter))
        return basic_masks, basic_masks_inter


class SipMaskHead(nn.Module):
    def __init__(self, cfg):
        super(SipMaskHead, self).__init__()
        self.num_basic_mask = cfg.MODEL.SIPMASK.HEAD.NUM_BASIC_MASKS
        self.num_sub_regions = cfg.MODEL.SIPMASK.HEAD.SUB_REGION_X * cfg.MODEL.SIPMASK.HEAD.SUB_REGION_Y
        self.in_channcels = cfg.MODEL.SIPMASK.HEAD.IN_CHANNELS
        self.deformable_groups = cfg.MODEL.SIPMASK.HEAD.DEFORMABLE_GROUPS
        self.fa_kernel_size = cfg.MODEL.SIPMASK.HEAD.FA_KERNEL_SIZE
        self.condition_loss_on = cfg.MODEL.SIPMASK.HEAD.CONDTION_INST_LOSS_ON
        self.out_stride = 8 # P3 stride

        self.feat_align = FeatureAlign(in_channels=self.in_channcels, out_channels=self.in_channcels,
                                       kernel_size=self.fa_kernel_size, deformable_groups=self.deformable_groups)

        self.cof_pred = CoefficientPredictor(in_channels=self.in_channcels, num_basic_mask=self.num_basic_mask,
                                             num_sub_regions=self.num_sub_regions)

        self.basic_mask_generator = BasicMaskGenerator(in_channels=self.in_channcels*3, channels=512,
                                                       num_basic_mask=self.num_basic_mask)

        self.feat_align.init_weights()

        if self.condition_loss_on:
            num_classes = cfg.MODEL.FCOS.NUM_CLASSES
            self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
            self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA
            norm = "SyncBN"
            conv_block = conv_with_kaiming_uniform(norm, True)
            in_channels = 512
            planes = 128 # with the same dimension as conditionInst
            self.seg_head = nn.Sequential(
                conv_block(in_channels, planes, kernel_size=3, stride=1),
                conv_block(planes, planes, kernel_size=3, stride=1)
            )

            self.logits = nn.Conv2d(planes, num_classes, kernel_size=1, stride=1)

            prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.logits.bias, bias_value)

    def forward(self, cls_towers, bbox_towers, bbox_preds, gt_instances=None):
        basic_masks = []
        aligned_cls_towers = []
        count = 0
        for cls_tower, bbox_tower, bbox_pred in zip(cls_towers, bbox_towers, bbox_preds):
            if not self.training and min(cls_tower.size()[2:]) < 3:
                o_size = cls_tower.size()[2:]
                h, w = cls_tower.size()[-2], cls_tower.size()[-1]
                if h < w:
                    ratio = w / h
                    cls_tower = F.interpolate(cls_tower, size=(3, int(3 * ratio)))
                    bbox_pred = F.interpolate(bbox_pred, size=(3, int(3 * ratio)))
                else:
                    ratio = h / w
                    cls_tower = F.interpolate(cls_tower, size=(int(3 * ratio), 3))
                    bbox_pred = F.interpolate(bbox_pred, size=(int(3 * ratio), 3))

            else:
                o_size = None
            aligned_cls_tower = self.feat_align(cls_tower, bbox_pred)

            if not self.training and o_size is not None:
                aligned_cls_tower = F.interpolate(aligned_cls_tower, size=(o_size[0], o_size[1]))

            aligned_cls_towers.append(aligned_cls_tower)

            if count < 3:
                feat_up = F.interpolate(bbox_tower, scale_factor=(2 ** count), mode='bilinear', align_corners=False)
                basic_masks.append(feat_up)
            count = count + 1

        cof_preds = self.cof_pred(aligned_cls_towers)

        losses = {}
        basic_masks = torch.cat(basic_masks, dim=1)
        basic_masks, basic_masks_inter = self.basic_mask_generator(basic_masks)

        if self.training and self.condition_loss_on:
            assert gt_instances is not None
            logits_pred = self.logits(self.seg_head(
                basic_masks_inter
            ))
            # compute semantic targets
            semantic_targets = []
            for per_im_gt in gt_instances:
                h, w = per_im_gt.gt_bitmasks_full.size()[-2:]
                areas = per_im_gt.gt_bitmasks_full.sum(dim=-1).sum(dim=-1)
                areas = areas[:, None, None].repeat(1, h, w)
                areas[per_im_gt.gt_bitmasks_full == 0] = INF
                areas = areas.permute(1, 2, 0).reshape(h * w, -1)
                min_areas, inds = areas.min(dim=1)
                per_im_sematic_targets = per_im_gt.gt_classes[inds] + 1
                per_im_sematic_targets[min_areas == INF] = 0
                per_im_sematic_targets = per_im_sematic_targets.reshape(h, w)
                semantic_targets.append(per_im_sematic_targets)

            semantic_targets = torch.stack(semantic_targets, dim=0)

            # resize target to reduce memory
            semantic_targets = semantic_targets[
                               :, None, self.out_stride // 2::self.out_stride,
                               self.out_stride // 2::self.out_stride
                               ]

            # prepare one-hot targets
            num_classes = logits_pred.size(1)
            class_range = torch.arange(
                num_classes, dtype=logits_pred.dtype,
                device=logits_pred.device
            )[:, None, None]
            class_range = class_range + 1
            one_hot = (semantic_targets == class_range).float()
            num_pos = (one_hot > 0).sum().float().clamp(min=1.0)

            loss_sem = sigmoid_focal_loss_jit(
                logits_pred, one_hot,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            ) / num_pos
            losses['loss_sem'] = loss_sem

        basic_masks = F.interpolate(basic_masks, scale_factor=4, mode='bilinear', align_corners=False)

        return aligned_cls_towers, cof_preds, basic_masks, losses


class SipMask(nn.Sequential):
    """
    Module for SipMask computation. Take feature maps from the backbone and
    SipMask outputs and losses
    """

    def __init__(self, cfg):
        super(SipMask, self).__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = cfg.build_backbone(cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
        self.proposal_generator = cfg.build_proposal_generator(cfg, self.backbone.output_shape())
        self.sip_head = build_sipmask_head(cfg)
        self.crop_size = cfg.MODEL.SIPMASK.HEAD.SUB_REGION_X
        self.crop_cuda = CropSplit(self.crop_size)
        self.crop_gt_cuda = CropSplitGT(self.crop_size)
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.sipmask_loss = build_sipmask_loss_computation(cfg)
        self.test_inference = build_sipmask_inference(cfg)
        self.mask_thresh = cfg.MODEL.SIPMASK.HEAD.MASK_THRESH
        self._init_track(cfg)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def _init_track(self, cfg):
        self.track_on = cfg.MODEL.TRACK_ON
        if not self.track_on:
            return
        else:
            self.track_head = build_track_head(cfg)

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if self.track_on:
            images_reference = [x["image_reference"].to(self.device) for x in batched_inputs]
            images_reference = [self.normalizer(x) for x in images_reference]
            images_reference = ImageList.from_tensors(images_reference, self.backbone.size_divisibility)
            features_reference = self.backbone(images_reference.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            self.add_bitmasks(gt_instances, images.tensor.size(-2), images.tensor.size(-1), self.mask_format)
        else:
            gt_instances = None

        cls_towers, bbox_tower, bbox_preds, ctrness_preds = self.proposal_generator.tower_generator(features)

        aligned_cls_towers, cof_preds, basic_masks, losses = self.sip_head(cls_towers, bbox_tower, bbox_preds, gt_instances)

        logits_pred, locations = self.proposal_generator(features, aligned_cls_towers, bbox_preds, ctrness_preds)

        if self.track_on:
            reference_gt_boxes = [x["instances_reference"].gt_boxes for x in batched_inputs]
            query_track, reference_track = self.track_head(features, features_reference)
            losses_sipmask = self.sipmask_loss.losses_with_track(logits_pred, bbox_preds, ctrness_preds, locations,
                                                         cof_preds, basic_masks, query_track, reference_track,
                                                         gt_instances, reference_gt_boxes)
        else:
            losses_sipmask = self.sipmask_loss(logits_pred, bbox_preds, ctrness_preds, locations,
                                       cof_preds, basic_masks, gt_instances)

        losses.update(losses_sipmask)

        return losses

    def inference(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)
        is_first = batched_inputs[0].get("is_first", None)

        cls_towers, bbox_tower, bbox_preds, ctrness_preds = self.proposal_generator.tower_generator(features)
        aligned_cls_towers, cof_preds, basic_masks, losses = self.sip_head(cls_towers, bbox_tower, bbox_preds)
        proposals, _ = self.proposal_generator(features, aligned_cls_towers, bbox_preds, ctrness_preds)
        padded_im_h, padded_im_w = images.tensor.size()[-2:]
        logits_pred = proposals["logits_pred"]
        reg_pred = proposals["reg_pred"]
        ctrness_pred = proposals["ctrness_pred"]
        features_inference = [features[f] for f in self.proposal_generator.in_features]
        locations = self.proposal_generator.compute_locations(features_inference)
        if self.track_on:
            query_track = self.track_head(features)
            boxeslists = self.test_inference.inference_with_track(locations, logits_pred, reg_pred, ctrness_pred,
                                                                  cof_preds, basic_masks, query_track,
                                                                  images.image_sizes, is_first)
        else:
            boxeslists = self.test_inference(locations, logits_pred, reg_pred, ctrness_pred, cof_preds,
                                            basic_masks, images.image_sizes)
        processed_results = []
        for boxeslist, image_size, input_image in zip(boxeslists, images.image_sizes, batched_inputs):
            height = input_image.get("height", image_size[0])
            width = input_image.get("width", image_size[1])
            boxeslist = self.postprocess(boxeslist, height, width, padded_im_h,
                                         padded_im_w, mask_threshold=self.mask_thresh)
            processed_results.append({
                "instances": boxeslist
            })
        return processed_results

    def _forward_mask_heads_train(self, proposals, cof_preds, basic_masks, gt_instances):

        pred_instances = proposals["instances"]
        if len(pred_instances) == 0:
            return 0.0
        num_images = len(gt_instances)

        pos_inds = pred_instances.pos_inds
        gt_inds = pred_instances.gt_inds
        flatten_cof_preds = [
            cof_pred.permute(0, 2, 3, 1).reshape(-1, self.sip_head.num_basic_mask * self.sip_head.num_sub_regions)
            for cof_pred in cof_preds
        ]  # [N, 32*4, H, W] -> [N, H, W, 32*4] -> [N*H*W, 32*4]
        flatten_cof_preds = cat(flatten_cof_preds, dim=0)                                         # [L*N*H*W, 32*4]
        flatten_cof_preds = flatten_cof_preds[pos_inds]

        reg_preds = pred_instances.reg_pred  # [num_pos, 4]
        fpn_levels = pred_instances.fpn_levels
        reg_preds = 2 ** (fpn_levels + 3)[:, None] * reg_preds
        locations = pred_instances.locations  # [num_pos, 2]
        detections = torch.stack([
            locations[:, 0] - reg_preds[:, 0],
            locations[:, 1] - reg_preds[:, 1],
            locations[:, 0] + reg_preds[:, 2],
            locations[:, 1] + reg_preds[:, 3]
        ], dim=1).detach()  # [num_pos, 4]

        area = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
        detections = detections[area > 1.0]
        flatten_cof_preds = flatten_cof_preds[area > 1.0]
        gt_inds = gt_inds[area > 1.0]

        gt_bboxes = cat([per_img.gt_boxes.tensor for per_img in gt_instances], dim=0)             # [num_box, 4]
        gt_bboxes = gt_bboxes[gt_inds]                                                            # [num_pos, 4]

        gt_masks = cat([per_img.gt_masks for per_img in gt_instances], dim=0)
        gt_masks = F.interpolate(gt_masks.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
        gt_masks = gt_masks[gt_inds]

        gt_labels = pred_instances.labels                                                        # [num_pos, ]
        gt_labels = gt_labels[area > 1.0]
        logits_pred = pred_instances.logits_pred                                                 # [num_pos, C]
        logits_pred = logits_pred[area > 1.0]
        scores = logits_pred[torch.arange(logits_pred.size(0)), gt_labels].sigmoid().detach()
        im_inds = pred_instances.im_inds
        im_inds = im_inds[area > 1.0]
        keeps = self._per_image_nms(im_inds, num_images, scores, detections)

        loss_mask = 0
        for i, keep in enumerate(keeps):
            detection = detections[keep]/2
            gt_bbox = gt_bboxes[keep]/2
            score = scores[keep]
            iou = align_box_iou_tensor(gt_bbox, detection)
            weighting = score * iou
            weighting = weighting / torch.sum(weighting) * len(weighting)
            flatten_cof_pred = flatten_cof_preds[keep]                                           # [keep, 32*4]
            gt_bit_mask = gt_masks[keep]
            gt_bit_mask = gt_bit_mask.gt(0.5).float().permute(1, 2, 0).contiguous()
            basic_mask = basic_masks[i].permute(1, 2, 0)                                          # [h, w, 32]
            num_basic_masks = basic_mask.size(2)
            pos_mask00 = torch.sigmoid(basic_mask @ flatten_cof_pred[:, 0:num_basic_masks].t())
            pos_mask01 = torch.sigmoid(basic_mask @ flatten_cof_pred[:, num_basic_masks:2*num_basic_masks].t())
            pos_mask10 = torch.sigmoid(basic_mask @ flatten_cof_pred[:, 2*num_basic_masks:3*num_basic_masks].t())
            pos_mask11 = torch.sigmoid(basic_mask @ flatten_cof_pred[:, 3*num_basic_masks:4*num_basic_masks].t())

            pred_masks = torch.stack([pos_mask00, pos_mask01, pos_mask10, pos_mask11], dim=0)       # [4, h, w, keep]
            gt_mask_crop = self.crop_gt_cuda(gt_bit_mask, detection)
            pred_masks = self.crop_cuda(pred_masks, detection)

            pre_losses = F.binary_cross_entropy(pred_masks, gt_mask_crop, reduction='none')
            centerwh = center_size(detection)
            pos_box_width = centerwh[:, 2]
            pos_box_height = centerwh[:, 3]
            pre_losses = pre_losses.sum(dim=(0, 1)) / pos_box_width / pos_box_height / centerwh.shape[0]
            loss_mask = loss_mask + torch.sum(pre_losses * weighting.detach())

        loss_mask = loss_mask / num_images
        if loss_mask > 1.0:
            loss_mask = loss_mask*0.5
        return loss_mask

    def _corner2center(self, boxes):
        return cat(( (boxes[:, 2:] + boxes[:, :2])/2,
                      boxes[:, 2:] - boxes[:, :2] ), dim=1)

    def _per_image_nms(self, im_inds, num_images, scores, detections):
        keeps = []
        for i in range(num_images):
            img_idx = torch.where(im_inds == i)[0]
            keeps.append(img_idx[torchvision.ops.nms(detections[img_idx], scores[img_idx], iou_threshold=0.9)])

        return keeps

    def add_bitmasks(self, instances, im_h, im_w, mask_format):
        if mask_format == 'polygon':
            for per_im_gt_inst in instances:
                if not per_im_gt_inst.has("gt_masks"):
                    continue
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_image_bitmasks = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    per_image_bitmasks.append(bitmask)

                gt_masks = torch.stack(per_image_bitmasks, dim=0)
                per_im_gt_inst.gt_masks = gt_masks
                gt_masks_full = gt_masks.clone()
                per_im_gt_inst.gt_bitmasks_full = gt_masks_full
        else:
            for per_im_gt_inst in instances:
                if not per_im_gt_inst.has("gt_masks"):
                    continue
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                bitmasks_full = bitmasks.clone()
                per_im_gt_inst.gt_masks = bitmasks
                per_im_gt_inst.gt_bitmasks_full = bitmasks_full

    def postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.4):
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size
        results = Instances((output_height, output_width), **results.get_fields())

        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

        results = results[output_boxes.nonempty()]
        if results.has("pred_global_masks"):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_w
            pred_global_masks = aligned_bilinear(results.pred_global_masks, factor)
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            results.pred_masks = (pred_global_masks > mask_threshold).float()

        return results


class SipMaskInference(nn.Module):
    def __init__(self, cfg):
        super(SipMaskInference, self).__init__()
        self.cfg = cfg
        self.pre_nms_thresh = cfg.MODEL.SIPMASK.INFERENCE_TH
        self.pre_nms_top_n = cfg.MODEL.SIPMASK.PRE_NMS_TOP_N
        self.nms_thresh = cfg.MODEL.SIPMASK.NMS_TH
        self.post_nms_top_n = cfg.MODEL.SIPMASK.POST_NMS_TOP_N
        self.thresh_with_ctr = cfg.MODEL.SIPMASK.THRESH_WITH_CTR
        self.strides = cfg.MODEL.SIPMASK.FPN_STRIDES
        self.num_sub_regions = cfg.MODEL.SIPMASK.HEAD.SUB_REGION_X * cfg.MODEL.SIPMASK.HEAD.SUB_REGION_Y
        self.num_basic_mask = cfg.MODEL.SIPMASK.HEAD.NUM_BASIC_MASKS
        self.crop_cuda = CropSplit(cfg.MODEL.SIPMASK.HEAD.SUB_REGION_X)
        if cfg.MODEL.TRACK_ON:
            self.prev_roi_feats = None
            self.prev_bboxes = None
            self.prev_det_labels = None
            self.match_coef = cfg.MODEL.SIPMASK.TRACKHEAD.MATCH_COEFF

    def forward(self, locations, logits_pred, reg_pred, ctrness_pred, det_cofs, basic_masks, image_sizes):

        sampled_boxes = []

        bundle = {
            "l": locations, "o": logits_pred,
            "r": reg_pred, "c": ctrness_pred,
            "d": det_cofs, "s": self.strides
        }
        for i, per_bundle in enumerate(zip(*bundle.values())):
            per_bundle = dict(zip(bundle.keys(), per_bundle))
            l = per_bundle["l"]
            o = per_bundle["o"]
            r = per_bundle["r"] * per_bundle["s"]
            c = per_bundle["c"]
            d = per_bundle["d"]
            sampled_boxes.append(self.forward_for_single_feature_map(l, o, r, c, d, image_sizes))

            for per_im_samples_bpxes in sampled_boxes[-1]:
                per_im_samples_bpxes.fpn_levels = l.new_ones(
                    len(per_im_samples_bpxes), dtype=torch.long
                ) * i

        boxlists = list(zip(*sampled_boxes))    # image first
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists, basic_masks)

        return boxlists

    def inference_with_track(self, locations, logits_pred, reg_pred, ctrness_pred, det_cofs,
                             basic_masks, query_track, image_sizes, is_first):
        sampled_boxes = []

        self.pre_nms_thresh = self.cfg.MODEL.SIPMASK.INFERENCE_TH_TRACK
        self.pre_nms_top_n = self.cfg.MODEL.SIPMASK.PRE_NMS_TOP_N_TRACK
        self.nms_thresh = self.cfg.MODEL.SIPMASK.NMS_TH_TRACK
        self.post_nms_top_n = self.cfg.MODEL.SIPMASK.POST_NMS_TOP_N_TRACK

        bundle = {
            "l": locations, "o": logits_pred,
            "r": reg_pred, "c": ctrness_pred,
            "d": det_cofs, "s": self.strides
        }
        for i, per_bundle in enumerate(zip(*bundle.values())):
            per_bundle = dict(zip(bundle.keys(), per_bundle))
            l = per_bundle["l"]
            o = per_bundle["o"]
            r = per_bundle["r"] * per_bundle["s"]
            c = per_bundle["c"]
            d = per_bundle["d"]
            sampled_boxes.append(self.forward_for_single_feature_map(l, o, r, c, d, image_sizes))

            for per_im_samples_boxes in sampled_boxes[-1]:
                per_im_samples_boxes.fpn_levels = l.new_ones(
                    len(per_im_samples_boxes), dtype=torch.long
                ) * i

        boxlists = list(zip(*sampled_boxes))  # image first
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists, basic_masks)

        for img_id in range(len(boxlists)):
            det_bboxes = boxlists[img_id].pred_boxes.tensor
            det_roi_feats = self.extract_box_feature_center_single(query_track[img_id], det_bboxes)
            det_labels = boxlists[img_id].pred_classes
            det_scores = boxlists[img_id].scores
            if det_bboxes.size(0) == 0:
                boxlists[img_id].pred_obj_ids = torch.ones((det_bboxes.shape[0]), dtype=torch.int) * (-1)
                continue
            if is_first or (not is_first and self.prev_bboxes is None):
                det_obj_ids = torch.arange(det_bboxes.size(0))
                self.prev_bboxes = det_bboxes
                self.prev_roi_feats = det_roi_feats
                self.prev_det_labels = det_labels
                boxlists[img_id].pred_obj_ids = det_obj_ids
            else:
                assert self.prev_roi_feats is not None
                prod = torch.mm(det_roi_feats, torch.transpose(self.prev_roi_feats, 0, 1))
                m = prod.size(0)
                dummy = torch.zeros(m, 1, device=torch.cuda.current_device())
                match_score = cat([dummy, prod], dim=1)
                match_logprob = F.log_softmax(match_score, dim=1)
                label_delta = (self.prev_det_labels == det_labels.view(-1, 1)).float()
                bbox_ious = pairwise_iou_tensor(det_bboxes, self.prev_bboxes)

                comp_scores = self.compute_comp_scores(match_logprob,
                                                       det_scores.view(-1, 1),
                                                       bbox_ious,
                                                       label_delta,
                                                       add_bbox_dummy=True)
                match_likelihood, match_ids = torch.max(comp_scores, dim=1)
                match_ids = match_ids.cpu().numpy().astype(np.int32)
                det_obj_ids = torch.ones((match_ids.shape[0]), dtype=torch.int) * (-1)
                best_match_scores = np.ones((self.prev_bboxes.size(0))) * (-100)
                for idx, match_id in enumerate(match_ids):
                    if match_id == 0:
                        det_obj_ids[idx] = self.prev_roi_feats.size(0)
                        self.prev_roi_feats = cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                        self.prev_bboxes = cat((self.prev_bboxes, det_bboxes[idx][None]), dim=0)
                        self.prev_det_labels = cat((self.prev_det_labels, det_labels[idx][None]), dim=0)
                    else:
                        obj_id = match_id - 1
                        match_score = comp_scores[idx, match_id]
                        if match_score > best_match_scores[obj_id]:
                            det_obj_ids[idx] = obj_id
                            best_match_scores[obj_id] = match_score
                            self.prev_roi_feats[obj_id] = det_roi_feats[idx]
                            self.prev_bboxes[obj_id] = det_bboxes[idx]
                boxlists[img_id].pred_obj_ids = det_obj_ids
        return boxlists

    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=False):
        if add_bbox_dummy:
            dummy_iou = torch.ones(bbox_ious.size(0), 1, device=torch.cuda.current_device()) * 0
            bbox_ious = cat([dummy_iou, bbox_ious], dim=1)
            dummy_label = torch.ones(bbox_ious.size(0), 1, device=torch.cuda.current_device())
            label_delta = cat([dummy_label, label_delta], dim=1)
        if self.match_coef is None:
            return match_ll
        else:
            assert len(self.match_coef) == 3
            return match_ll + self.match_coef[0] * torch.log(bbox_scores) + \
                   self.match_coef[1] * bbox_ious + self.match_coef[2] * label_delta

    def extract_box_feature_center_single(self, track_feats, gt_bboxes):
        track_box_feats = track_feats.new_zeros(gt_bboxes.size(0), 512)

        ref_feat_stride = 8
        gt_center_xs = torch.floor((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0 / ref_feat_stride).long()
        gt_center_ys = torch.floor((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0 / ref_feat_stride).long()

        aa = track_feats.permute(1, 2, 0)          # [h,w,512]
        bb = aa[gt_center_ys, gt_center_xs, :]
        track_box_feats += bb
        return track_box_feats

    def forward_for_single_feature_map(self, locations, logits_pred, reg_pred,
                                       ctrness_pred, det_cofs, image_sizes):
        N, C, H, W = logits_pred.shape

        # put in the same format as locations
        logits_pred = logits_pred.view(N, C, H, W).permute(0, 2, 3, 1)
        logits_pred = logits_pred.reshape(N, -1, C).sigmoid()
        reg_pred = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        reg_pred = reg_pred.reshape(N, -1, 4)
        ctrness_pred = ctrness_pred.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness_pred = ctrness_pred.reshape(N, -1).sigmoid()
        det_cofs = det_cofs.view(N, self.num_basic_mask*self.num_sub_regions, H, W).permute(0, 2, 3, 1)
        det_cofs = det_cofs.reshape(N, -1, self.num_basic_mask*self.num_sub_regions)

        if self.thresh_with_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]
        candidate_inds = logits_pred > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        if not self.thresh_with_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = logits_pred[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()       # n, 2
            per_box_loc = per_candidate_nonzeros[:, 0]                      # which loation
            per_class = per_candidate_nonzeros[:, 1]                        # which class

            per_box_regression = reg_pred[i]
            per_box_regression = per_box_regression[per_box_loc]        # according bbox prediction
            per_locations = locations[per_box_loc]                      # according locations

            per_det_cofs = det_cofs[i]
            per_det_cofs = per_det_cofs[per_box_loc]                    # according cof predictions


            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                per_det_cofs = per_det_cofs[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            boxlist.det_cofs = per_det_cofs

            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists, basic_masks):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            if number_of_detections > self.post_nms_top_n > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]

            det_cofs = result.det_cofs                  # keep, num_basic*num_sub_regions
            det_bboxes = result.pred_boxes.tensor

            if len(result) > 0:
                scale = 2

                basic_mask = basic_masks[i].permute(1, 2, 0)          # h, w, num_basic_masks
                pos_masks00 = torch.sigmoid(basic_mask @ det_cofs[:, 0:self.num_basic_mask].t())   # h, w, keep
                pos_masks01 = torch.sigmoid(basic_mask @ det_cofs[:, self.num_basic_mask:2*self.num_basic_mask].t())
                pos_masks10 = torch.sigmoid(basic_mask @ det_cofs[:, 2*self.num_basic_mask:3*self.num_basic_mask].t())
                pos_masks11 = torch.sigmoid(basic_mask @ det_cofs[:, 3*self.num_basic_mask:4*self.num_basic_mask].t())
                pos_masks = torch.stack([pos_masks00, pos_masks01, pos_masks10, pos_masks11], dim=0)   # [4, h, w, keep]
                pos_masks = self.crop_cuda(pos_masks, det_bboxes/scale).permute(2, 0, 1)                   # [keep, h, w]
                masks = F.interpolate(pos_masks.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)
                masks = masks[:, None]
            else:
                masks = det_bboxes.new_empty((0, 1, result.image_size[0], result.image_size[1]))

            result.pred_global_masks = masks
            results.append(result)
        return results

    def fast_nms(self, boxes, scores, masks, cfg, iou_threshold=0.5, top_k=200):
        scores, idx = scores.sort(1, descending=True)

        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]

        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = self.jaccard(boxes, boxes)
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        # Now just filter out the ones higher than the threshold
        keep = (iou_max <= iou_threshold)

        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.
        keep *= (scores > cfg.score_thr)

        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]

        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_per_img]
        scores = scores[:cfg.max_per_img]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]
        boxes = torch.cat([boxes, scores[:, None]], dim=1)
        return boxes, classes, masks

    def jaccard(self, box_a, box_b, iscrowd: bool = False):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """
        use_batch = True
        if box_a.dim() == 2:
            use_batch = False
            box_a = box_a[None, ...]
            box_b = box_b[None, ...]

        inter = self.intersect(box_a, box_b)
        area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
                  (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
                  (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter

        out = inter / area_a if iscrowd else inter / union
        return out if use_batch else out.squeeze(0)


class SipMaskLossComputation(object):

    def __init__(self, cfg):
        self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA
        self.center_sample = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.radius = cfg.MODEL.FCOS.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.FCOS.INFERENCE_TH_TRAIN
        self.pre_nms_topk_train = cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN
        self.post_nms_topk_train = cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN
        self.loc_loss_func = IOULoss(cfg.MODEL.FCOS.LOC_LOSS_TYPE)
        self.amplitude = cfg.MODEL.SIPMASK.TRACKHEAD.AMPLITUDE

        self.num_basic_mask = cfg.MODEL.SIPMASK.HEAD.NUM_BASIC_MASKS
        self.num_sub_regions = cfg.MODEL.SIPMASK.HEAD.SUB_REGION_X * cfg.MODEL.SIPMASK.HEAD.SUB_REGION_Y

        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.strides = cfg.MODEL.FCOS.FPN_STRIDES

        self.crop_cuda = CropSplit(cfg.MODEL.SIPMASK.HEAD.SUB_REGION_X)
        self.crop_gt_cuda = CropSplitGT(cfg.MODEL.SIPMASK.HEAD.SUB_REGION_X)

        soi = []
        prev_size = -1
        for s in cfg.MODEL.FCOS.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi

    def get_sample_region(self, boxes, strides, num_loc_list, loc_xs, loc_ys, bitmasks=None, radius=1):
        if bitmasks is not None:
            _, h, w = bitmasks.size()

            ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

            m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
            m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_x = m10 / m00
            center_y = m01 / m00
        else:
            center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
            center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5

        num_gts = boxes.shape[0]
        K = len(loc_xs)
        boxes = boxes[None].expand(K, num_gts, 4)
        center_x = center_x[None].expand(K, num_gts)
        center_y = center_y[None].expand(K, num_gts)
        center_gt = boxes.new_zeros(boxes.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, size_ranges, num_loc_list):
        labels = []
        reg_targets = []
        gt_inds = []
        gt_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                gt_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                if targets_per_im.has("gt_masks_full"):
                    bitmasks = targets_per_im.gt_masks_full
                else:
                    bitmasks = None
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, num_loc_list, xs, ys,
                    bitmasks=bitmasks, radius=self.radius
                )
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            num_targets += len(targets_per_im)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            gt_target = bboxes[locations_to_gt_inds]

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            gt_inds.append(locations_to_gt_inds[labels_per_im != self.num_classes])
            gt_targets.append(gt_target)

        return {
            "labels": labels,
            "reg_targets": reg_targets,
            "gt_inds": gt_inds,
            "gt_targets": gt_targets
        }

    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first

    def _get_ground_truth(self, locations, gt_instances):
        num_loc_list = [len(loc) for loc in locations]
        self.num_points_per_level = num_loc_list

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(locations, dim=0)

        training_targets = self.compute_targets_for_locations(
            locations, gt_instances, loc_to_size_range, num_loc_list
        )

        # transpose im first training_targets to level first ones

        labels_level_first = []
        reg_targets_level_first = []

        labels = training_targets["labels"]
        reg_targets = training_targets["reg_targets"]
        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_loc_list, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_loc_list, dim=0)
        for level in range(len(num_loc_list)):
            labels_level_first.append(
                cat([label_per_im[level] for label_per_im in labels], dim=0)
            )
            reg_targets_per_level = cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            reg_targets_per_level = reg_targets_per_level / float(self.strides[level])
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first, labels, training_targets["gt_targets"], training_targets["gt_inds"]

    def _decode_for_single_feature_map(self, locations, box_regression, fpn_stride):
        N, A, H, W = box_regression.shape

        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4) * fpn_stride
        results = []
        for i in range(N):
            per_box_regression = box_regression[i]
            per_locations = locations

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)
            results.append(detections)

        return results

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, logits_pred, reg_pred, ctrness_pred, locations, cof_preds, basic_masks, gt_instances):

        num_classes = logits_pred[0].size(1)
        labels, reg_targets, label_list, box_gt_list, gt_inds = self._get_ground_truth(locations, gt_instances)

        ###decode boxes###
        sampled_boxes = []
        for _, (l, b, s) in enumerate(zip(locations, reg_pred, self.strides)):
            sampled_boxes.append(self._decode_for_single_feature_map(l, b, s))

        flatten_sampled_boxes = [
            torch.cat([labels_level_img.reshape(-1, 4)
                       for labels_level_img in sampled_boxes_per_img])
            for sampled_boxes_per_img in zip(*sampled_boxes)
        ]

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(logits_pred[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(reg_pred[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(ctrness_pred[l].reshape(-1))

        box_cls_flatten = cat(box_cls_flatten, dim=0)
        box_regression_flatten = cat(box_regression_flatten, dim=0)
        centerness_flatten = cat(centerness_flatten, dim=0)
        labels_flatten = cat(labels_flatten, dim=0)
        reg_targets_flatten = cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten != num_classes, as_tuple=False).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        class_target = torch.zeros_like(box_cls_flatten)
        class_target[pos_inds, labels_flatten[pos_inds]] = 1

        cls_loss = sigmoid_focal_loss_jit(
            box_cls_flatten,
            class_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            reg_loss = self.loc_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            ) / sum_centerness_targets_avg_per_gpu

            centerness_loss = F.binary_cross_entropy_with_logits(
                centerness_flatten,
                centerness_targets,
                reduction="sum"
            ) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum() * 0
            centerness_loss = centerness_flatten.sum() * 0

        ###mask loss###
        num_imgs = len(flatten_sampled_boxes)
        flatten_cls_scores1 = []
        for l in range(len(labels)):
            flatten_cls_scores1.append(logits_pred[l].permute(0, 2, 3, 1).reshape(num_imgs, -1, num_classes))

        flatten_cls_scores1 = cat(flatten_cls_scores1, dim=1)

        flatten_cof_preds = [
            cof_pred.permute(0, 2, 3, 1).reshape(len(label_list), -1, self.num_basic_mask*self.num_sub_regions)
            for cof_pred in cof_preds
        ]

        flatten_cof_preds = cat(flatten_cof_preds, dim=1)

        loss_mask = 0
        for i in range(num_imgs):
            labels = torch.cat([labels_level.flatten() for labels_level in label_list[i]])

            bbox_dt = flatten_sampled_boxes[i] / 2
            bbox_dt = bbox_dt.detach()
            pos_inds = labels != num_classes

            cof_pred = flatten_cof_preds[i][pos_inds]
            img_mask = basic_masks[i]
            mask_h = img_mask.shape[1]
            mask_w = img_mask.shape[2]
            idx_gt = gt_inds[i]
            bbox_dt = bbox_dt[pos_inds, :4]
            gt_masks = gt_instances[i].gt_masks.to(dtype=torch.float32, device=img_mask.device)
            gt_masks = gt_masks.reshape(-1, gt_masks.shape[-2], gt_masks.shape[-1])

            area = (bbox_dt[:, 2] - bbox_dt[:, 0]) * (bbox_dt[:, 3] - bbox_dt[:, 1])
            bbox_dt = bbox_dt[area > 1.0, :]
            idx_gt = idx_gt[area > 1.0]
            cof_pred = cof_pred[area > 1.0]
            if bbox_dt.shape[0] == 0:
                continue

            bbox_gt = gt_instances[i].gt_boxes.tensor
            cls_score = flatten_cls_scores1[i, pos_inds, labels[pos_inds]].sigmoid().detach()
            cls_score = cls_score[area > 1.0]
            ious = align_box_iou_tensor(bbox_gt[idx_gt]/2, bbox_dt)
            weighting = cls_score * ious
            weighting = weighting / torch.sum(weighting)*len(weighting)
            keep = torchvision.ops.nms(bbox_dt, cls_score, 0.9)
            bbox_dt = bbox_dt[keep]
            weighting = weighting[keep]
            idx_gt = idx_gt[keep]
            cof_pred = cof_pred[keep]

            gt_mask = F.interpolate(gt_masks.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)

            shape = np.minimum(basic_masks[i].shape, gt_mask.shape)
            gt_mask_new = gt_mask.new_zeros(gt_mask.shape[0], mask_h, mask_w)
            gt_mask_new[:gt_mask.shape[0], :shape[1], :shape[2]] = gt_mask[:gt_mask.shape[0], :shape[1], :shape[2]]
            gt_mask_new = gt_mask_new.gt(0.5).float()

            gt_mask_new = torch.index_select(gt_mask_new, 0, idx_gt).permute(1, 2, 0).contiguous()

            img_mask1 = img_mask.permute(1, 2, 0)
            pos_mask00 = torch.sigmoid(img_mask1 @ cof_pred[:, 0:self.num_basic_mask].t())
            pos_mask01 = torch.sigmoid(img_mask1 @ cof_pred[:, self.num_basic_mask:2*self.num_basic_mask].t())
            pos_mask10 = torch.sigmoid(img_mask1 @ cof_pred[:, 2*self.num_basic_mask:3*self.num_basic_mask].t())
            pos_mask11 = torch.sigmoid(img_mask1 @ cof_pred[:, 3*self.num_basic_mask:4*self.num_basic_mask].t())
            pred_masks = torch.stack([pos_mask00, pos_mask01, pos_mask10, pos_mask11], dim=0)
            pred_masks = self.crop_cuda(pred_masks, bbox_dt)
            gt_mask_crop = self.crop_gt_cuda(gt_mask_new, bbox_dt)
            pre_loss = F.binary_cross_entropy(pred_masks, gt_mask_crop, reduction='none')
            pos_get_csize = center_size(bbox_dt)
            gt_box_width = pos_get_csize[:, 2]
            gt_box_height = pos_get_csize[:, 3]
            pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height / pos_get_csize.shape[0]
            loss_mask += torch.sum(pre_loss * weighting.detach())

        loss_mask = loss_mask / num_imgs
        if loss_mask > 1.0:
            loss_mask = loss_mask * 0.5
        if loss_mask == 0:
            loss_mask = flatten_cof_preds.sum() * 0

        losses = {
            "loss_fcos_cls": cls_loss,
            "loss_fcos_reg": reg_loss,
            "loss_fcos_centerness": centerness_loss,
            "loss_sipmask_mask": loss_mask
        }
        return losses

    def _extract_box_feature_center_single(self, track_feats, gt_boxes):
        track_box_feats = track_feats.new_zeros(gt_boxes.size()[0], 512)

        ref_feat_stride = 8
        gt_center_xs = torch.floor((gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0 / ref_feat_stride).long()        # n,
        gt_center_ys = torch.floor((gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0 / ref_feat_stride).long()        # n,

        aa = track_feats.permute(1, 2, 0)               # h, w, 512
        bb = aa[gt_center_ys, gt_center_xs, :]          # n, 512
        track_box_feats += bb

        return track_box_feats

    def losses_with_track(self,
                          logits_pred,
                          reg_pred,
                          ctrness_pred,
                          locations,
                          cof_preds,
                          basic_masks,
                          query_track,
                          reference_track,
                          gt_instances,
                          reference_gt_boxes):

        assert len(logits_pred) == len(reg_pred) == len(ctrness_pred)                   # len = L
        num_classes = logits_pred[0].size(1)
        labels, reg_targets, label_list, box_gt_list, gt_inds = self._get_ground_truth(locations, gt_instances)

        ###decode boxes###
        sampled_boxes = []
        for _, (l, b, s) in enumerate(zip(locations, reg_pred, self.strides)):      # level first
            sampled_boxes.append(self._decode_for_single_feature_map(l, b, s))

        flatten_sampled_boxes = [
            torch.cat([labels_level_img.reshape(-1, 4)
                       for labels_level_img in sampled_boxes_per_img])
            for sampled_boxes_per_img in zip(*sampled_boxes)
        ]                                                                       # image first

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(logits_pred[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(reg_pred[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(ctrness_pred[l].reshape(-1))

        box_cls_flatten = cat(box_cls_flatten, dim=0)
        box_regression_flatten = cat(box_regression_flatten, dim=0)
        centerness_flatten = cat(centerness_flatten, dim=0)
        labels_flatten = cat(labels_flatten, dim=0)
        reg_targets_flatten = cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten != num_classes, as_tuple=False).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        class_target = torch.zeros_like(box_cls_flatten)
        class_target[pos_inds, labels_flatten[pos_inds]] = 1

        cls_loss = sigmoid_focal_loss_jit(
            box_cls_flatten,
            class_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            reg_loss = self.loc_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            ) / sum_centerness_targets_avg_per_gpu

            centerness_loss = F.binary_cross_entropy_with_logits(
                centerness_flatten,
                centerness_targets,
                reduction="sum"
            ) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum() * 0
            centerness_loss = centerness_flatten.sum() * 0

        ###mask loss###
        num_imgs = len(flatten_sampled_boxes)
        flatten_cls_scores1 = []
        for l in range(len(labels)):
            flatten_cls_scores1.append(logits_pred[l].permute(0, 2, 3, 1).reshape(num_imgs, -1, num_classes))

        flatten_cls_scores1 = cat(flatten_cls_scores1, dim=1)

        flatten_cof_preds = [
            cof_pred.permute(0, 2, 3, 1).reshape(len(label_list), -1, self.num_basic_mask * self.num_sub_regions)
            for cof_pred in cof_preds
        ]

        flatten_cof_preds = cat(flatten_cof_preds, dim=1)

        loss_mask = 0
        loss_match = 0
        n_total = 0
        for i in range(num_imgs):
            labels = torch.cat([labels_level.flatten() for labels_level in label_list[i]])

            bbox_dt = flatten_sampled_boxes[i] / 2
            bbox_dt = bbox_dt.detach()
            pos_inds = labels != num_classes

            cof_pred = flatten_cof_preds[i][pos_inds]
            img_mask = basic_masks[i]
            mask_h = img_mask.shape[1]
            mask_w = img_mask.shape[2]
            idx_gt = gt_inds[i]
            bbox_dt = bbox_dt[pos_inds, :4]
            gt_masks = gt_instances[i].gt_masks.to(dtype=torch.float32, device=img_mask.device)
            gt_masks = gt_masks.reshape(-1, gt_masks.shape[-2], gt_masks.shape[-1])

            area = (bbox_dt[:, 2] - bbox_dt[:, 0]) * (bbox_dt[:, 3] - bbox_dt[:, 1])
            bbox_dt = bbox_dt[area > 1.0, :]
            idx_gt = idx_gt[area > 1.0]
            cof_pred = cof_pred[area > 1.0]
            if bbox_dt.shape[0] == 0:
                loss_mask += area.sum() * 0
                continue

            bbox_gt = gt_instances[i].gt_boxes.tensor
            cls_score = flatten_cls_scores1[i, pos_inds, labels[pos_inds]].sigmoid().detach()
            cls_score = cls_score[area > 1.0]
            ious = align_box_iou_tensor(bbox_gt[idx_gt]/2, bbox_dt)
            weighting = cls_score * ious
            weighting = weighting / torch.sum(weighting + 1e-4) * len(weighting)

            ####################track#######################
            bboxes_ref = reference_gt_boxes[i].tensor
            random_offsets = bboxes_ref.new_empty(bboxes_ref.shape[0], 4).uniform_(
                -self.amplitude, self.amplitude)                            # n, 4
            # before jittering
            cxcy = (bboxes_ref[:, 2:4] + bboxes_ref[:, :2]) / 2
            wh = (bboxes_ref[:, 2:4] - bboxes_ref[:, :2]).abs()             # [n, 2]
            # after jittering
            new_cxcy = cxcy + wh * random_offsets[:, :2]
            new_wh = wh * (1 + random_offsets[:, 2:])
            # cxcywh to xyxy
            new_x1y1 = new_cxcy - new_wh / 2
            new_x2y2 = new_cxcy + new_wh / 2
            new_bboxes = cat([new_x1y1, new_x2y2], dim=1)
            query_track_feat = self._extract_box_feature_center_single(query_track[i], bbox_dt * 2)     # number_pos, 512
            reference_track_feat = self._extract_box_feature_center_single(reference_track[i], new_bboxes)     # num_reference, 512
            gt_pids = gt_instances[i].gt_pids
            cur_ids = gt_pids[idx_gt]
            prod = torch.mm(query_track_feat, torch.transpose(reference_track_feat, 0, 1))              # num_pos, num_reference
            m = prod.size(0)
            dummy = torch.zeros(m, 1, device=torch.cuda.current_device())

            prod_ext = cat([dummy, prod], dim=1)
            loss_match += F.cross_entropy(prod_ext, cur_ids, reduction='mean')
            n_total += len(idx_gt)
        #    match_acc += accuracy(prod_ext, cur_ids) * len(idx_gt)

            gt_mask = F.interpolate(gt_masks.unsqueeze(0), scale_factor=0.5,
                                    mode='bilinear', align_corners=False).squeeze(0)

            shape = np.minimum(basic_masks[i].shape, gt_mask.shape)
            gt_mask_new = gt_mask.new_zeros(gt_mask.shape[0], mask_h, mask_w)
            gt_mask_new[:gt_mask.shape[0], :shape[1], :shape[2]] = gt_mask[:gt_mask.shape[0], :shape[1], :shape[2]]
            gt_mask_new = gt_mask_new.gt(0.5).float()

            gt_mask_new = torch.index_select(gt_mask_new, 0, idx_gt).permute(1, 2, 0).contiguous()

            img_mask1 = img_mask.permute(1, 2, 0)
            pos_mask00 = torch.sigmoid(img_mask1 @ cof_pred[:, 0:self.num_basic_mask].t())
            pos_mask01 = torch.sigmoid(img_mask1 @ cof_pred[:, self.num_basic_mask:2*self.num_basic_mask].t())
            pos_mask10 = torch.sigmoid(img_mask1 @ cof_pred[:, 2*self.num_basic_mask:3*self.num_basic_mask].t())
            pos_mask11 = torch.sigmoid(img_mask1 @ cof_pred[:, 3*self.num_basic_mask:4*self.num_basic_mask].t())
            pred_masks = torch.stack([pos_mask00, pos_mask01, pos_mask10, pos_mask11], dim=0)
            pred_masks = self.crop_cuda(pred_masks, bbox_dt)
            gt_mask_crop = self.crop_gt_cuda(gt_mask_new, bbox_dt)
            pre_loss = F.binary_cross_entropy(pred_masks, gt_mask_crop, reduction='none')
            pos_get_csize = center_size(bbox_dt)
            gt_box_width = pos_get_csize[:, 2]
            gt_box_height = pos_get_csize[:, 3]
            pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height / pos_get_csize.shape[0]
            loss_mask += torch.sum(pre_loss * weighting.detach())

        loss_mask = loss_mask / num_imgs
        loss_match = loss_match / num_imgs
     #   match_acc = match_acc / n_total
        if loss_mask > 1.0:
            loss_mask = loss_mask * 0.5
        if loss_mask == 0:
            loss_mask = bbox_dt[:, 0].sum() * 0

        losses = {
            "loss_fcos_cls": cls_loss,
            "loss_fcos_reg": reg_loss,
            "loss_fcos_centerness": centerness_loss,
            "loss_sipmask_mask": loss_mask,
            "loss_match": loss_match
        }
        return losses


def build_sipmask_loss_computation(cfg):

    loss_computation = SipMaskLossComputation(cfg)
    return loss_computation


def build_sipmask_inference(cfg):

    box_selector = SipMaskInference(cfg)
    return box_selector


def build_track_head(cfg):

    sipmask_trackhead = SipMaskTrackHead(cfg)
    return sipmask_trackhead


def build_sipmask_head(cfg):

    return SipMaskHead(cfg)
