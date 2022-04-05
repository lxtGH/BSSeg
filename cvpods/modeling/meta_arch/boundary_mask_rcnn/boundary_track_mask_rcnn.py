
from typing import Dict, List, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from cvpods.layers import Conv2d, ConvTranspose2d, cat, get_norm

from cvpods.utils import get_event_storage
from cvpods.modeling.roi_heads.mask_head import mask_rcnn_inference
from cvpods.layers import ShapeSpec
from cvpods.structures import Instances
from cvpods.modeling.roi_heads import StandardROIHeads
from cvpods.modeling.poolers import ROIPooler
from cvpods.modeling.roi_heads.roi_heads import select_foreground_proposals
from cvpods.modeling.roi_heads.cascade_rcnn import CascadeROIHeads

class BoundaryROIHeads(StandardROIHeads):

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_mask_head(cfg)

    def _init_mask_head(self, cfg):
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # edge poolers
        boundary_resolution     = cfg.MODEL.BOUNDARY_MASK_HEAD.POOLER_RESOLUTION
        boundary_in_features    = cfg.MODEL.BOUNDARY_MASK_HEAD.IN_FEATURES
        boundary_scales         = tuple(1.0 / self.feature_strides[k] for k in boundary_in_features)
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_in_features = in_features
        self.boundary_in_features = boundary_in_features

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.boundary_pooler = ROIPooler(
            output_size=boundary_resolution,
            scales=boundary_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type
        )
        self.mask_head = cfg.build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        mask_features = [features[i] for i in range(len(self.mask_in_features))]
        boundary_features = [features[i] for i in range(len(self.boundary_in_features))]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(mask_features, proposal_boxes)
            boundary_features = self.boundary_pooler(boundary_features, proposal_boxes)
            return self.mask_head(mask_features, boundary_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(mask_features, pred_boxes)
            boundary_features = self.boundary_pooler(boundary_features, pred_boxes)
            return self.mask_head(mask_features, boundary_features, instances)


class CascadeBoundaryROIHeads(CascadeROIHeads):

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_mask_head(cfg)

    def _init_mask_head(self, cfg):
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # edge poolers
        boundary_resolution     = cfg.MODEL.BOUNDARY_MASK_HEAD.POOLER_RESOLUTION
        boundary_in_features    = cfg.MODEL.BOUNDARY_MASK_HEAD.IN_FEATURES
        boundary_scales         = tuple(1.0 / self.feature_strides[k] for k in boundary_in_features)
        # fmt: on

        in_channels = [self.feature_channels[f] for f in in_features][0]

        self.mask_in_features = in_features
        self.boundary_in_features = boundary_in_features
        # ret = {"mask_in_features": in_features}
        self.mask_pooler= ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.boundary_pooler = ROIPooler(
            output_size=boundary_resolution,
            scales=boundary_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type
        )
        self.mask_head = cfg.build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        mask_features = [features[f] for f in range(len(self.mask_in_features))]
        boundary_features = [features[f] for f in range(len(self.boundary_in_features))]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(mask_features, proposal_boxes)
            boundary_features = self.boundary_pooler(boundary_features, proposal_boxes)
            return self.mask_head(mask_features, boundary_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(mask_features, pred_boxes)
            boundary_features = self.boundary_pooler(boundary_features, pred_boxes)
            return self.mask_head(mask_features, boundary_features, instances)


def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


def boundary_loss_func(boundary_logits, gtmasks):
    """
    Args:
        boundary_logits (Tensor): A tensor of shape (B, H, W) or (B, H, W)
        gtmasks (Tensor): A tensor of shape (B, H, W) or (B, H, W)
    """
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=boundary_logits.device).reshape(1, 1, 3, 3).requires_grad_(False)
    boundary_logits = boundary_logits.unsqueeze(1)
    boundary_targets = F.conv2d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0

    if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
        boundary_targets = F.interpolate(
            boundary_targets, boundary_logits.shape[2:], mode='nearest')

    bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
    dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boundary_targets)
    return bce_loss + dice_loss


def boundary_preserving_mask_loss(
        pred_mask_logits,
        pred_boundary_logits,
        instances,
        vis_period=0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0, pred_boundary_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
        pred_boundary_logits = pred_boundary_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]
        pred_boundary_logits = pred_boundary_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    boundary_loss = boundary_loss_func(pred_boundary_logits, gt_masks)
    return mask_loss, boundary_loss


def combined_boundary_mask_loss(
        pred_mask_logits,
        pred_mask_in_logits,
        pred_mask_out_logits,
        coarse_pred_boundary_logits,
        fine_pred_boundary_logits,
        instances,
        vis_period=0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0, coarse_pred_boundary_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
        pred_mask_in_logits = pred_mask_in_logits[:, 0]
        pred_mask_out_logits = pred_mask_out_logits[:, 0]
        coarse_pred_boundary_logits = coarse_pred_boundary_logits[:, 0]
        fine_pred_boundary_logits = fine_pred_boundary_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]
        pred_mask_in_logits = pred_mask_in_logits[indices, gt_classes]
        pred_mask_out_logits = pred_mask_out_logits[indices, gt_classes]
        coarse_pred_boundary_logits = coarse_pred_boundary_logits[indices, gt_classes]
        fine_pred_boundary_logits = fine_pred_boundary_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    mask_loss_in = F.binary_cross_entropy_with_logits(pred_mask_in_logits, gt_masks, reduction="mean")
    mask_loss_out = F.binary_cross_entropy_with_logits(pred_mask_out_logits, gt_masks, reduction="mean")
    boundary_loss_coarse = boundary_loss_func(coarse_pred_boundary_logits, gt_masks)
    boundary_loss_fine = boundary_loss_func(fine_pred_boundary_logits, gt_masks)
    return mask_loss, mask_loss_in, mask_loss_out, boundary_loss_coarse, boundary_loss_fine


class BoundaryPreservingHead(nn.Module):

    def __init__(self, cfg, input_shape: ShapeSpec):
        super(BoundaryPreservingHead, self).__init__()

        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        conv_norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_boundary_conv = cfg.MODEL.BOUNDARY_MASK_HEAD.NUM_CONV
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            num_classes = 1

        self.mask_fcns = []
        cur_channels = input_shape.channels
        for k in range(num_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.mask_fcns.append(conv)
            cur_channels = conv_dim

        self.mask_final_fusion = Conv2d(
            conv_dim, conv_dim,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu)

        self.downsample = Conv2d(
            conv_dim, conv_dim,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )
        self.boundary_fcns = []
        cur_channels = input_shape.channels
        for k in range(num_boundary_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("boundary_fcn{}".format(k + 1), conv)
            self.boundary_fcns.append(conv)
            cur_channels = conv_dim

        self.mask_to_boundary = Conv2d(
            conv_dim, conv_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )

        self.boundary_to_mask = Conv2d(
            conv_dim, conv_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )

        self.mask_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )
        self.mask_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        self.boundary_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )
        self.boundary_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.mask_fcns + self.boundary_fcns +\
                     [self.mask_deconv, self.boundary_deconv, self.boundary_to_mask, self.mask_to_boundary,
                      self.mask_final_fusion, self.downsample]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.mask_predictor.weight, std=0.001)
        nn.init.normal_(self.boundary_predictor.weight, std=0.001)
        if self.mask_predictor.bias is not None:
            nn.init.constant_(self.mask_predictor.bias, 0)
        if self.boundary_predictor.bias is not None:
            nn.init.constant_(self.boundary_predictor.bias, 0)

    def forward(self, mask_features, boundary_features, instances: List[Instances]):
        for layer in self.mask_fcns:
            mask_features = layer(mask_features)
        # downsample
        boundary_features = self.downsample(boundary_features)
        # mask to boundary fusion
        boundary_features = boundary_features + self.mask_to_boundary(mask_features)
        for layer in self.boundary_fcns:
            boundary_features = layer(boundary_features)
        # boundary to mask fusion
        mask_features = self.boundary_to_mask(boundary_features) + mask_features
        mask_features = self.mask_final_fusion(mask_features)
        # mask prediction
        mask_features = F.relu(self.mask_deconv(mask_features))
        mask_logits = self.mask_predictor(mask_features)
        # boundary prediction
        boundary_features = F.relu(self.boundary_deconv(boundary_features))
        boundary_logits = self.boundary_predictor(boundary_features)
        if self.training:
            loss_mask, loss_boundary = boundary_preserving_mask_loss(
                mask_logits, boundary_logits, instances)
            return {"loss_mask": loss_mask,
                    "loss_boundary": loss_boundary}
        else:
            mask_rcnn_inference(mask_logits, instances)
            return instances


class BoundaryFlowRefine(nn.Module):
    def __init__(self, inplane, norm_cfg, act_cfg):
        """implementation of body generation part
        """
        super(BoundaryFlowRefine, self).__init__()
        self.smooth = nn.Sequential(
            Conv2d(inplane, inplane, 3, stride=2, padding=1, groups=inplane,
                 norm=get_norm(norm_cfg, inplane), activation=act_cfg),
            Conv2d(inplane, inplane, 3, stride=2, padding=1, groups=inplane,
                 norm=get_norm(norm_cfg, inplane), activation=act_cfg)
        )
        self.flow_make_in2out = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)
        self.flow_make_out2in = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x, coarse_edge):
        # 形态学收缩
        x_body_out = x - coarse_edge
        flow_in = self.flow_make_out2in(torch.cat([x_body_out, x], dim=1))
        x_warp_in = self.flow_warp(x, flow_in, size=flow_in.size()[2:])
        # 形态学膨胀
        x_body_in = self.smooth(x)
        seg_down = F.upsample(x_body_in, size=x.size()[2:], mode="bilinear", align_corners=True)
        flow_out = self.flow_make_in2out(torch.cat([seg_down, x], dim=1))
        x_warp_out = self.flow_warp(seg_down, flow_out, size=x.size()[2:])
        # 收缩和膨胀相交于更加精细的边缘
        f_edge_fine = x_warp_in - x_warp_out

        return x_warp_in, x_warp_out, f_edge_fine

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


class DecoupledBoundaryMaskHead(nn.Module):

    def __init__(self, cfg, input_shape: ShapeSpec):
        super(DecoupledBoundaryMaskHead, self).__init__()

        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        conv_norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        output_ratio = cfg.MODEL.BOUNDARY_MASK_HEAD.OUTPUT_RATIO
        num_boundary_conv = cfg.MODEL.BOUNDARY_MASK_HEAD.NUM_CONV
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            num_classes = 1

        self.mask_fcns = []
        cur_channels = input_shape.channels
        for k in range(num_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.mask_fcns.append(conv)
            cur_channels = conv_dim

        self.mask_final_fusion = Conv2d(
            conv_dim, conv_dim,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu)

        self.downsample = Conv2d(
            conv_dim, conv_dim,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )
        self.boundary_fcns = []
        cur_channels = input_shape.channels
        self.output_ratio = output_ratio
        for k in range(num_boundary_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("boundary_fcn{}".format(k + 1), conv)
            self.boundary_fcns.append(conv)
            cur_channels = conv_dim

        self.mask_to_boundary = Conv2d(
            conv_dim, conv_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )

        self.boundary_to_mask = Conv2d(
            conv_dim, conv_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )

        self.mask_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )
        self.mask_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        self.boundary_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )
        self.boundary_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        # extra boundary supervision
        self.fine_boundary_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        self.bf_module = BoundaryFlowRefine(conv_dim, norm_cfg="BN", act_cfg=F.relu)

        # extra mask supervision

        self.mask_predictor_in = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.mask_predictor_out = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.mask_fcns + self.boundary_fcns +\
                     [self.mask_deconv, self.boundary_deconv, self.boundary_to_mask, self.mask_to_boundary,
                      self.mask_final_fusion, self.downsample]:
            weight_init.c2_msra_fill(layer)

        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.mask_predictor.weight, std=0.001)
        nn.init.normal_(self.boundary_predictor.weight, std=0.001)
        nn.init.normal_(self.fine_boundary_predictor.weight, std=0.001)
        nn.init.normal_(self.mask_predictor_in.weight, std=0.001)
        nn.init.normal_(self.mask_predictor_out.weight, std=0.001)

        if self.mask_predictor.bias is not None:
            nn.init.constant_(self.mask_predictor.bias, 0)
        if self.boundary_predictor.bias is not None:
            nn.init.constant_(self.boundary_predictor.bias, 0)

    def forward(self, mask_features, boundary_features, instances: List[Instances]):
        for layer in self.mask_fcns:
            mask_features = layer(mask_features)
        if mask_features.shape[0] == 0:
            return instances
        # downsample
        boundary_features = self.downsample(boundary_features)
        # mask to boundary fusion
        boundary_features = boundary_features + self.mask_to_boundary(mask_features)
        for layer in self.boundary_fcns:
            boundary_features = layer(boundary_features)
        # boundary to mask fusion

        # coarse boundary prediction
        boundary_features = F.relu(self.boundary_deconv(boundary_features))

        coarse_boundary_logits = self.boundary_predictor(boundary_features)

        mask_features = self.mask_final_fusion(mask_features)
        # mask features
        mask_features = F.relu(self.mask_deconv(mask_features))

        b, c, h, w = mask_features.size()

        if self.output_ratio > 1:
            mask_features = F.upsample(mask_features, size=(h*self.output_ratio, w*self.output_ratio), mode="bilinear", align_corners=True)
            boundary_features = F.upsample(mask_features, size=(h*self.output_ratio, w*self.output_ratio), mode="bilinear", align_corners=True)
            coarse_boundary_logits = F.upsample(coarse_boundary_logits, size=(h*self.output_ratio, w*self.output_ratio), mode="bilinear", align_corners=True)

        mask_in, mask_out, boundary_fine = self.bf_module(mask_features, boundary_features)
        mask_features = self.boundary_to_mask(boundary_fine) + mask_in + mask_out

        # mask prediction
        mask_logits = self.mask_predictor(mask_features)

        if self.training:
            # extra loss
            mask_logits_in = self.mask_predictor_in(mask_in)
            mask_logits_out = self.mask_predictor_out(mask_out)
            fine_boundary_logits = self.fine_boundary_predictor(boundary_fine)

            loss_mask, loss_mask_in, loss_mask_out, loss_coarse_boundary, loss_fine_boundary = combined_boundary_mask_loss(
                mask_logits, mask_logits_in, mask_logits_out, coarse_boundary_logits, fine_boundary_logits, instances)
            return {
                    "loss_mask": loss_mask * 0.9,
                    "loss_mask_in": loss_mask_in * 0.05,
                    "loss_mask_out": loss_mask_out * 0.05,
                    "loss_coarse_boundary": loss_coarse_boundary * 0.1,
                    "loss_fine_boundary": loss_fine_boundary * 0.8
                    }
        else:
            mask_rcnn_inference(mask_logits, instances)
            return instances