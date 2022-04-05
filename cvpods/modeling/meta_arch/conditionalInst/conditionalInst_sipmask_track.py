"""
Condition Inst Re-implementation using cvpods
Xiangtai Li(lixiangtai@sensetime.com)
"""
# -*- coding: utf-8 -*-
import logging
import math

import torch
from torch import nn
from typing import Dict
import torch.nn.functional as F
import numpy as np

from cvpods.structures import ImageList, pairwise_iou_tensor
from cvpods.structures.instances import Instances
from cvpods.structures.masks import polygons_to_bitmask
from cvpods.modeling.losses import sigmoid_focal_loss_jit
from cvpods.layers import ShapeSpec, NaiveSyncBatchNorm, NaiveGroupNorm, cat
from cvpods.layers.deform_conv import DFConv2d
from cvpods.layers.conv_with_kaiming_uniform import conv_with_kaiming_uniform
from cvpods.layers.misc import aligned_bilinear, compute_locations


logger = logging.getLogger(__name__)
INF = 100000000


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def build_mask_branch(cfg, input_shape):
    return MaskBranch(cfg, input_shape)


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


def build_tracking_head(cfg):
    return CondInstTrackingHead(cfg)


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


class CondInstTrackingHead(nn.Module):
    def __init__(self, cfg):
        super(CondInstTrackingHead, self).__init__()
        self.stacked_convs = cfg.MODEL.CONDINST.TRACKHEAD.NUM_TRACK_CONVS
        self.use_deformable = cfg.MODEL.CONDINST.TRACKHEAD.USE_DEFORMABLE
        self.in_channels = cfg.MODEL.CONDINST.TRACKHEAD.IN_CHANNELS
        self.feat_channels = cfg.MODEL.CONDINST.TRACKHEAD.FEAT_CHANNELS
        self.track_feat_channels = cfg.MODEL.CONDINST.TRACKHEAD.TRACK_FEAT_CHANNELS
        self.norm = None if cfg.MODEL.CONDINST.TRACKHEAD.NORM == 'none' else cfg.MODEL.CONDINST.TRACKHEAD.NORM
        self.in_features = cfg.MODEL.CONDINST.TRACKHEAD.IN_FEATURES
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
                                   stride=1,
                                   padding=1,
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

        self.track_pred = nn.Conv2d(self.feat_channels*len(self.in_features), self.track_feat_channels,
                                    kernel_size=1, padding=0)

    def _train_forward(self, query_feats, reference_feats):
        query_track_feats = []
        reference_track_feats = []
        count = 0
        assert len(query_feats) == len(reference_feats)
        for query_feat, reference_feat in zip(query_feats, reference_feats):
            if count < 3:
                query_track_feat = self.track_tower(query_feat)
                query_track_feat = F.interpolate(query_track_feat, scale_factor=2**count,
                                                 mode='bilinear', align_corners=False)
                query_track_feats.append(query_track_feat)
                reference_track_feat = self.track_tower(reference_feat)
                reference_track_feat = F.interpolate(reference_track_feat, scale_factor=2**count,
                                                     mode='bilinear', align_corners=False)
                reference_track_feats.append(reference_track_feat)
            else:
                break

            count += 1

        query_track_feats = cat(query_track_feats, dim=1)
        reference_track_feats = cat(reference_track_feats, dim=1)

        query_track = self.track_pred(query_track_feats)
        reference_track = self.track_pred(reference_track_feats)

        return query_track, reference_track

    def _inference_forward(self, query_feats):
        query_track_feats = []
        count = 0
        for query_feat in query_feats:
            if count < 3:
                query_track_feat = self.track_tower(query_feat)
                query_track_feat = F.interpolate(query_track_feat, scale_factor=2**count,
                                                 mode='bilinear', align_corners=False)
                query_track_feats.append(query_track_feat)
            else:
                break
            count += 1
        query_track_feats = cat(query_track_feats, dim=1)
        query_track = self.track_pred(query_track_feats)

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


class MaskBranch(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES
        self.sem_loss_on = cfg.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON
        self.num_outputs = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS

        norm = cfg.MODEL.CONDINST.MASK_BRANCH.NORM
        num_convs = cfg.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS
        channels = cfg.MODEL.CONDINST.MASK_BRANCH.CHANNELS
        self.out_stride = input_shape[self.in_features[0]].stride

        feature_channels = {k: v.channels for k, v in input_shape.items()}

        conv_block = conv_with_kaiming_uniform(norm, activation=True)

        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(
                feature_channels[in_feature],
                channels, 3, 1
            ))

        tower = []
        for i in range(num_convs):
            tower.append(conv_block(
                channels, channels, 3, 1
            ))

        self.add_module('tower', nn.Sequential(*tower))
        self.out_fea = nn.Conv2d(
            channels, max(self.num_outputs, 1), 1
        )
        if self.sem_loss_on:
            num_classes = cfg.MODEL.FCOS.NUM_CLASSES
            self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
            self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA

            in_channels = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(
                conv_block(in_channels, channels, kernel_size=3, stride=1),
                conv_block(channels, channels, kernel_size=3, stride=1)
            )

            self.logits = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)

            prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.logits.bias, bias_value)

    def forward(self, features, gt_instances=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])

                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p

        mask_feats = self.tower(x)

        mask_feats = self.out_fea(mask_feats)

        if self.num_outputs == 0:
            mask_feats = mask_feats[:, :self.num_outputs]

        losses = {}
        # auxiliary thing semantic loss
        if self.training and self.sem_loss_on:
            logits_pred = self.logits(self.seg_head(
                features[self.in_features[0]]
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

        return mask_feats, losses


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations                    # num_pos, 2
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)    # [num_pos, 1, 2] - [1, num_ponits_s8, 2] = [num_pos, num_points_s8, 2]
            relative_coords = relative_coords.permute(0, 2, 1).float()  # [num_pos, 2, num_points_s8]
            soi = self.sizes_of_interest.float()[instances.fpn_levels]  # [num_pos, 2]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)   # [num_pos, 2, num_points_s8]/[num_pos, 2, 1] = [num_pos, 2, num_points_r8]
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)
            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)           # [num_pos, 10, h*w]

        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)  # (1, b*c,h,w)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)     # [1, num_pos, H, W]

        mask_logits = mask_logits.reshape(-1, 1, H, W)              # [num_pos, 1, H, W]

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits.sigmoid()

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training:
            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            if len(pred_instances) == 0:
                loss_mask = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
            else:
                mask_scores = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
                loss_mask = mask_losses.mean()

            return loss_mask.float()
        else:
            if len(pred_instances) > 0:
                mask_scores = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_scores.float()

            return pred_instances


class Controller(nn.Module):
    def __init__(self, in_channels, kernel):
        super(Controller, self).__init__()
        self.in_channels = in_channels
        self.kernel = kernel
        self.conv = nn.Conv2d(in_channels, kernel, kernel_size=3, stride=1, padding=1)

        torch.nn.init.normal_(self.conv.weight, std=0.01)
        torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class CondInst(nn.Module):
    """
    Main class for CondInst architectures (see https://arxiv.org/abs/2003.05664).
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = cfg.build_backbone(cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
        self.proposal_generator = cfg.build_proposal_generator(cfg, self.backbone.output_shape())
        self.mask_head = build_dynamic_mask_head(cfg)
        self.mask_branch = build_mask_branch(cfg, self.backbone.output_shape())
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.max_proposals = cfg.MODEL.CONDINST.MAX_PROPOSALS
        self.mask_format = cfg.INPUT.MASK_FORMAT

        # build top module
        in_channels = self.proposal_generator.in_channels_to_top_module

        # build track head
        self._init_track_head(cfg)

        self.controller = Controller(
            in_channels, self.mask_head.num_gen_params)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def _init_track_head(self, cfg):
        if not cfg.MODEL.TRACK_ON:
            return
        else:
            self.track_head = build_tracking_head(cfg)
            self.amplitude = cfg.MODEL.CONDINST.TRACKHEAD.AMPLITUDE
            self.prev_roi_feats = None
            self.prev_bboxes = None
            self.prev_det_labels = None
            self.match_coef = cfg.MODEL.CONDINST.TRACKHEAD.MATCH_COEFF

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            self.add_bitmasks(gt_instances, images.tensor.size(-2), images.tensor.size(-1), self.mask_format)
        else:
            gt_instances = None

        mask_feats, sem_losses = self.mask_branch(features, gt_instances)

        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances, self.controller
        )

        if self.training:
            reference_images = [x["image_reference"].to(self.device) for x in batched_inputs]
            reference_images = [self.normalizer(x) for x in reference_images]
            reference_images = ImageList.from_tensors(reference_images, self.backbone.size_divisibility)
            reference_features = self.backbone(reference_images.tensor)

            reference_gt_boxes = [x["instances_reference"].gt_boxes for x in batched_inputs]
            query_track, reference_track = self.track_head(features, reference_features)

            loss_mask = self._forward_mask_heads_train(proposals, mask_feats, gt_instances)
            loss_match = self._forward_track_head_train(proposals, query_track, reference_track,
                                                        gt_instances, reference_gt_boxes)

            losses = {}
            losses.update(sem_losses)
            losses.update(proposal_losses)
            losses.update(loss_match)
            losses.update({"loss_mask": loss_mask})
            return losses
        else:
            pred_instances_w_masks = self._forward_mask_heads_test(proposals, mask_feats)
            is_first = batched_inputs[0].get("is_first", None)
            query_track = self.track_head(features)

            padded_im_h, padded_im_w = images.tensor.size()[-2:]
            processed_results = []
            for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images.image_sizes)):         # Loop for each image
                height = input_per_image.get("height", image_size[0])                   # height before resize
                width = input_per_image.get("width", image_size[1])                     # width before resize

                instances_per_im = pred_instances_w_masks[pred_instances_w_masks.im_inds == im_id]
                if len(instances_per_im) == 0:
                    instances_per_im.pred_global_masks = instances_per_im.pred_boxes.tensor.new_empty(0, 1,
                                                                        instances_per_im.image_size[0],
                                                                        instances_per_im.image_size[1])
                instances_per_im = self._forward_track_heads_test(instances_per_im, query_track, is_first)
                instances_per_im = self.postprocess(
                    instances_per_im, height, width,
                    padded_im_h, padded_im_w
                )

                processed_results.append({
                    "instances": instances_per_im
                })

            return processed_results

    def _forward_mask_heads_train(self, proposals, mask_feats, gt_instances):
        # prepare the inputs for mask heads
        pred_instances = proposals["instances"]

        if 0 <= self.max_proposals < len(pred_instances):
            inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
            logger.info("clipping proposals from {} to {}".format(
                len(pred_instances), self.max_proposals
            ))
            pred_instances = pred_instances[inds[:self.max_proposals]]

        pred_instances.mask_head_params = pred_instances.top_feats

        loss_mask = self.mask_head(
            mask_feats, self.mask_branch.out_stride,
            pred_instances, gt_instances
        )

        return loss_mask

    def _forward_mask_heads_test(self, proposals, mask_feats):
        # prepare the inputs for mask heads
        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        pred_instances = Instances.cat(proposals)
        pred_instances.mask_head_params = pred_instances.top_feat

        pred_instances_w_masks = self.mask_head(
            mask_feats, self.mask_branch.out_stride, pred_instances
        )

        return pred_instances_w_masks

    def _forward_track_heads_test(self, proposals, query_track, is_first):
        det_bboxes = proposals.pred_boxes.tensor
        det_labels = proposals.pred_classes
        det_scores = proposals.scores
        if det_bboxes.size(0) == 0:
            proposals.pred_obj_ids = torch.ones((det_bboxes.shape[0]), dtype=torch.int) * (-1)
            return proposals
        det_roi_feats = self._extract_box_feature_center_single(query_track[0], det_bboxes)
        if is_first or (not is_first and self.prev_bboxes is None):
            det_obj_ids = torch.arange(det_bboxes.size(0))
            self.prev_roi_feats = det_roi_feats
            self.prev_bboxes = det_bboxes
            self.prev_det_labels = det_labels
            proposals.pred_obj_ids = det_obj_ids
        else:
            assert self.prev_roi_feats is not None
            prod = torch.mm(det_roi_feats, torch.transpose(self.prev_roi_feats, 0, 1))
            n = prod.size(0)
            dummy = torch.zeros(n, 1, device=torch.cuda.current_device())
            match_score = cat([dummy, prod], dim=1)
            mat_logprob = F.log_softmax(match_score, dim=1)
            label_delta = (self.prev_det_labels == det_labels.view(-1, 1)).float()
            bbox_ious = pairwise_iou_tensor(det_bboxes, self.prev_bboxes)
            comp_scores = self.compute_comp_scores(mat_logprob,
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
            proposals.pred_obj_ids = det_obj_ids
        return proposals

    def _forward_track_head_train(self, proposals, query_track, reference_track, gt_instances, reference_gt_boxes):
        num_images = query_track.size(0)
        pred_instances = proposals["instances"]                 # level first, all images concat
        gt_pids = [x.gt_pids for x in gt_instances]

        if 0 <= self.max_proposals < len(pred_instances):
            inds = torch.randperm(len(pred_instances), device=query_track.device).long()
            pred_instances = pred_instances[inds[:self.max_proposals]]
        locations = pred_instances.locations                            # n, 2
        reg_pred = pred_instances.reg_pred                              # n, 4
        fpn_levels = pred_instances.fpn_levels                          # n,
        im_ids = pred_instances.im_inds                                 # n, fpn first
        gt_inds_relative = pred_instances.gt_inds_relative              # n,

        reg_pred = reg_pred * 2**(fpn_levels+3).view(-1, 1)
        detections = torch.stack([
            locations[:, 0] - reg_pred[:, 0],
            locations[:, 1] - reg_pred[:, 1],
            locations[:, 0] + reg_pred[:, 2],
            locations[:, 1] + reg_pred[:, 3]
        ], dim=1)                                                       # n,4

        loss_match = 0
        n_total = 0
        for i in range(num_images):
            instance_index_this_image = torch.where(im_ids == i)[0]     # which instance belong to this image
            if len(instance_index_this_image) == 0:                     # no instance
                loss_match += detections[instance_index_this_image].sum() * 0
                continue
            detections_this_image = detections[instance_index_this_image]   # detection results belong to this image
            gt_pids_this_image = gt_pids[i]                             # gt pids of this image
            gt_inds_relative_this_image = gt_inds_relative[instance_index_this_image]
            gt_pids_this_image = gt_pids_this_image[gt_inds_relative_this_image]

            reference_boxes_this_image = reference_gt_boxes[i].tensor
            random_offset = reference_boxes_this_image.new_empty(reference_boxes_this_image.shape[0], 4).uniform_(
                -self.amplitude, self.amplitude
            )
            # before jittering
            cxcy = (reference_boxes_this_image[:, 2:4] + reference_boxes_this_image[:, :2]) / 2
            wh = (reference_boxes_this_image[:, 2:4] - reference_boxes_this_image[:, :2]).abs()
            # after jittering
            new_cxcy = cxcy + wh * random_offset[:, :2]
            new_wh = wh * (1 + random_offset[:, 2:])
            new_x1y1 = new_cxcy - new_wh / 2
            new_x2y2 = new_cxcy + new_wh / 2
            new_boxes = cat([new_x1y1, new_x2y2], dim=1)

            query_track_feat = self._extract_box_feature_center_single(query_track[i], detections_this_image) # [n, 512]
            reference_track_feat = self._extract_box_feature_center_single(reference_track[i], new_boxes)     # [m, 512]
            prod = torch.mm(query_track_feat, torch.transpose(reference_track_feat, 0, 1))
            n = prod.size(0)
            dummy = torch.zeros(n, 1, device=torch.cuda.current_device())
            prod_ext = cat([dummy, prod], dim=1)
            loss_match += F.cross_entropy(prod_ext, gt_pids_this_image, reduction='mean')
            n_total += len(gt_pids_this_image)

        loss_match = loss_match / num_images

        losses = {"loss_match": loss_match}

        return losses

    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=True):
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

    def _extract_box_feature_center_single(self, track_feats, boxes):
        track_box_feats = track_feats.new_zeros(boxes.size(0), self.track_head.track_feat_channels)

        ref_feat_stride = 8
        boxes_center_xs = torch.floor((boxes[:, 0] + boxes[:, 2]) / 2.0 / ref_feat_stride).long()
        boxes_center_ys = torch.floor((boxes[:, 1] + boxes[:, 3]) / 2.0 / ref_feat_stride).long()

        aa = track_feats.permute(1, 2, 0)
        bb = aa[boxes_center_ys, boxes_center_xs, :]
        track_box_feats += bb

        return track_box_feats

    def add_bitmasks(self, instances, im_h, im_w, mask_format):
        if mask_format == 'polygon':
            for per_im_gt_inst in instances:
                if not per_im_gt_inst.has("gt_masks"):
                    continue
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    start = int(self.mask_out_stride // 2)
                    bitmask_full = bitmask.clone()
                    bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                    assert bitmask.size(0) * self.mask_out_stride == im_h
                    assert bitmask.size(1) * self.mask_out_stride == im_w

                    per_im_bitmasks.append(bitmask)
                    per_im_bitmasks_full.append(bitmask_full)

                per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
        else:
            for per_im_gt_inst in instances:
                if not per_im_gt_inst.has("gt_masks"):
                    continue
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                pad_size = bitmasks.shape
                mask_h = pad_size[-2]
                mask_w = pad_size[-1]
                pad_masks = bitmasks.new_full((pad_size[0], im_h, im_w), 0)
                pad_masks[:, :mask_h, :mask_w] = bitmasks
                pad_masks_full = pad_masks.clone()
                start = int(self.mask_out_stride // 2)
                pad_masks = pad_masks[:, start::self.mask_out_stride, start::self.mask_out_stride]
                per_im_gt_inst.gt_bitmasks = pad_masks
                per_im_gt_inst.gt_bitmasks_full = pad_masks_full


    def postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.5):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model, based on the output resolution
        """
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size                         # shape after resize before padding
        results = Instances((output_height, output_width), **results.get_fields())

        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)           # shape after resize before padding

        results = results[output_boxes.nonempty()]

        if results.has("pred_global_masks"):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(
                results.pred_global_masks, factor
            )
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            results.pred_masks = (pred_global_masks > mask_threshold).float()

        return results