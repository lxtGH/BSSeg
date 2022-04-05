
from typing import Dict, List, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from cvpods.layers import Conv2d, ConvTranspose2d, cat, get_norm

from cvpods.utils import get_event_storage
from cvpods.modeling.roi_heads.mask_head import mask_rcnn_inference
from cvpods.layers import ShapeSpec, NaiveSyncBatchNorm, NaiveGroupNorm, DeformConv, ModulatedDeformConv
from cvpods.structures import Instances
from cvpods.modeling.roi_heads import StandardROIHeads
from cvpods.modeling.poolers import ROIPooler
from cvpods.modeling.roi_heads.roi_heads import select_foreground_proposals
import cv2
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
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES                     # p2, p3, p4, p5
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION           #14
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # edge poolers
        boundary_resolution     = cfg.MODEL.BOUNDARY_MASK_HEAD.POOLER_RESOLUTION        # 28
        boundary_in_features    = cfg.MODEL.BOUNDARY_MASK_HEAD.IN_FEATURES              # p2
        boundary_scales         = tuple(1.0 / self.feature_strides[k] for k in boundary_in_features)
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_in_features = in_features                                     # p2, p3, p4, p5
        self.boundary_in_features = boundary_in_features                        # p2 for accuracy location

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
            mask_features = self.mask_pooler(mask_features, proposal_boxes)             # M, C, 14, 14
            boundary_features = self.boundary_pooler(boundary_features, proposal_boxes)  # M, C, 28, 28
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

        mask_features = [features[f] for f in range(len(self.mask_in_features))]                # p2, p3, p4, p5
        boundary_features = [features[f] for f in range(len(self.boundary_in_features))]        # p2

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


def boundary_loss_func(boundary_logits, gtmasks, return_boundary_gt=False):
    """
    Args:
        boundary_logits (Tensor): A tensor of shape (B, H, W) or (B, H, W)
        gtmasks (Tensor): A tensor of shape (B, H, W) or (B, H, W)
    Returns:
        boundary_loss
        boundary_targets: A tensor of shape (B, 1, H, W)
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
    if return_boundary_gt:
        return bce_loss + dice_loss, boundary_targets
    else:
        return bce_loss + dice_loss, None

def contraction_expansion_loss_func(pred_contraction_logits,
                                    pred_expansion_logits,
                                    gt_masks_bool,
                                    boundary_masks,
                                    kernel_size=5):

    boundary_masks = boundary_masks.squeeze(1)
    gt_mask_numpy = gt_masks_bool.cpu().numpy().astype('uint8')
    boundary_mask_numpy = boundary_masks.cpu().numpy().astype('uint8')
    contraction_masks = []
    expansion_masks = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    for gt_mask, boundary_mask in zip(gt_mask_numpy, boundary_mask_numpy):

        boundary_valid = boundary_mask == 1
        dilate = cv2.dilate(gt_mask, kernel=kernel)
        contraction_mask = dilate - gt_mask
        contraction_mask[boundary_valid] = 1
        contraction_mask = torch.from_numpy(contraction_mask).to(pred_contraction_logits.device)
        contraction_masks.append(contraction_mask.float())

        erode = cv2.erode(gt_mask, kernel=kernel)
        expansion_mask = gt_mask - erode
        expansion_mask[boundary_valid] = 1
        expansion_mask = torch.from_numpy(expansion_mask).to(pred_contraction_logits.device)
        expansion_masks.append(expansion_mask.float())

    contraction_mask = torch.stack(contraction_masks, dim=0)
    expansion_mask = torch.stack(expansion_masks, dim=0)

    # contraction_bce_loss = F.binary_cross_entropy_with_logits(pred_contraction_logits, contraction_mask)
    contraction_dice_loss = dice_loss_func(torch.sigmoid(pred_contraction_logits), contraction_mask)

    # expansion_bce_loss = F.binary_cross_entropy_with_logits(pred_expansion_logits, expansion_mask)
    expansion_dice_loss = dice_loss_func(torch.sigmoid(pred_expansion_logits), expansion_mask)

    return contraction_dice_loss, expansion_dice_loss

def joint_loss(
        pred_mask_logits,
        pred_contraction_mask_logits,
        pred_expansion_mask_logits,
        pred_boundary_logits,
        instances,
        kernel_size=5,
        vis_period=0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_mask_in_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask), or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks.
            the mask logits predicted of the inner part.
        pred_mask_out_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask), or (B, 1, Hmask, Wmask)
            for class-specitic or class-agnostic, where B is the total number of predicted masks.
            the mask logits predicted of the outter part.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
        boundary_loss (Tensor): A scalar tensor containing the boundary loss.
        contraction_loss (Tensor): A scalar tensor containing the contraction part loss.
        expansion_loss (Tensor): A scalar tensor containing the expansion loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1  # number class
    total_num_masks = pred_mask_logits.size(0)  # M
    mask_side_len = pred_mask_logits.size(2)  # 14/28/56/......
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:  # Loop dor each images
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:  # if is class-specific
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0, pred_boundary_logits.sum() * 0, pred_contraction_mask_logits.sum() * 0, pred_expansion_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
        pred_boundary_logits = pred_boundary_logits[:, 0]
        pred_contraction_mask_logits = pred_contraction_mask_logits[:, 0]
        pred_expansion_mask_logits = pred_expansion_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]
        pred_boundary_logits = pred_boundary_logits[indices, gt_classes]
        pred_contraction_mask_logits = pred_contraction_mask_logits[indices, gt_classes]
        pred_expansion_mask_logits = pred_expansion_mask_logits[indices, gt_classes]

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
    boundary_loss, boundary_target = boundary_loss_func(pred_boundary_logits,
                                                        gt_masks,
                                                        return_boundary_gt=True)
    contraction_loss, expansion_loss = contraction_expansion_loss_func(pred_contraction_mask_logits,
                                                                       pred_expansion_mask_logits,
                                                                       gt_masks_bool,
                                                                       boundary_target,
                                                                       kernel_size=kernel_size)
    return mask_loss, boundary_loss, contraction_loss, expansion_loss

class ContractionExpansionModule(nn.Module):
    def __init__(self, cfg, in_channels):
        super(ContractionExpansionModule, self).__init__()
        self.num_conv = cfg.MODEL.ROI_MASK_HEAD.CEMODULE.NUM_CONV
        self.planes = cfg.MODEL.ROI_MASK_HEAD.CEMODULE.PLANES
        self.dcn_on = cfg.MODEL.ROI_MASK_HEAD.CEMODULE.DCN_ON
        self.dcn_v2 = cfg.MODEL.ROI_MASK_HEAD.CEMODULE.DCN_V2
        self.num_edge_conv = cfg.MODEL.ROI_MASK_HEAD.CEMODULE.NUM_EDGE_CONV
        self.fuse_mode = cfg.MODEL.ROI_MASK_HEAD.CEMODULE.FUSE_MODE
        self.with_edge_feat_refine = cfg.MODEL.ROI_MASK_HEAD.CEMODULE.WITH_EDGE_REFINE          # hack, in_channels must equals to planes
        norm = None if cfg.MODEL.ROI_MASK_HEAD.CEMODULE.NORM == 'GN' else cfg.MODEL.ROI_MASK_HEAD.CEMODULE.NORM
        contraction_tower = []
        expansion_tower = []
        for i in range(self.num_conv):
            if i == 0:
                in_planes = in_channels
            else:
                in_planes = self.planes
            out_planes = self.planes
            contraction_tower.append(nn.Conv2d(in_channels=in_planes,
                                               out_channels=out_planes,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               bias=False))
            expansion_tower.append(nn.Conv2d(in_channels=in_planes,
                                             out_channels=out_planes,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             bias=False))
            if i != self.num_conv - 1:
                if norm == "GN":
                    contraction_tower.append(nn.GroupNorm(32, out_planes))
                    expansion_tower.append(nn.GroupNorm(32, out_planes))
                elif norm == "NaiveGN":
                    contraction_tower.append(NaiveGroupNorm(32, out_planes))
                    expansion_tower.append(NaiveGroupNorm(32, out_planes))
                elif norm == "BN":
                    contraction_tower.append(nn.BatchNorm2d(out_planes))
                    expansion_tower.append(nn.BatchNorm2d(out_planes))
                elif norm == "SyncBN":
                    contraction_tower.append(NaiveSyncBatchNorm(out_planes))
                    expansion_tower.append(NaiveSyncBatchNorm(out_planes))

                contraction_tower.append(nn.ReLU())
                expansion_tower.append(nn.ReLU())

        self.add_module("contraction_tower", nn.Sequential(*contraction_tower))
        self.add_module("expansion_tower", nn.Sequential(*expansion_tower))

        self.flow_mask_contraction = nn.Conv2d(self.planes * 2, 2, kernel_size=3, padding=1, bias=False)  # 内部的膨胀
        self.flow_mask_expansion = nn.Conv2d(self.planes * 2, 2, kernel_size=3, padding=1, bias=False)  # 外部的收缩

        edge_tower = []
        for i in range(self.num_edge_conv):
            if i == 0:
                if self.fuse_mode == "Add":
                    in_planes = self.planes
                elif self.fuse_mode == "Sub":
                    in_planes = self.planes
                elif self.fuse_mode == "Concat":
                    in_planes = 2 * self.planes + in_channels
            else:
                in_planes = self.planes
            out_planes = self.planes
            edge_tower.append(nn.Conv2d(in_channels=in_planes,
                                        out_channels=out_planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False))

            if i != self.num_edge_conv - 1:
                if norm == "GN":
                    edge_tower.append(nn.GroupNorm(32, out_planes))
                elif norm == "NaiveGN":
                    edge_tower.append(NaiveGroupNorm(32, out_planes))
                elif norm == "BN":
                    edge_tower.append(nn.BatchNorm2d(out_planes))
                elif norm == "SyncBN":
                    edge_tower.append(NaiveSyncBatchNorm(out_planes))

                edge_tower.append(nn.ReLU())

        self.add_module("edge_tower", nn.Sequential(*edge_tower))

        for modules in [self.contraction_tower, self.expansion_tower, self.edge_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

        for modules in [self.contraction_tower, self.expansion_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

    def forward(self, mask_features, boundary_features=None):
        contraction_features = mask_features.clone().detach() + boundary_features
        expansion_features = mask_features.clone().detach() + boundary_features
        contraction_features = self.contraction_tower(contraction_features)
        expansion_features = self.expansion_tower(expansion_features)

        contraction_warp = self.flow_mask_contraction(cat([contraction_features, mask_features], dim=1))
        expansion_warp = self.flow_mask_expansion(cat([expansion_features, mask_features], dim=1))

        contraction_features = self.flow_warp(contraction_features, contraction_warp,
                                              size=contraction_features.size()[-2:])
        expansion_features = self.flow_warp(expansion_features, expansion_warp, size=expansion_features.size()[-2:])

        boundary_features = contraction_features + expansion_features + mask_features
        boundary_features = self.edge_tower(boundary_features)
        return boundary_features, contraction_features, expansion_features

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
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.kernel_size = cfg.MODEL.ROI_MASK_HEAD.CEMODULE.KERNEL_SIZE
        self.loss_weight = cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT
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
        )                       # 下采样2倍
        cur_channels = input_shape.channels

        self.boundary_to_mask = Conv2d(
            conv_dim, conv_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )  # boundary 到 mask 转移的路径

        self.mask_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )  # mask的上采样
        self.mask_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)  # 预测mask的卷积核

        self.boundary_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )  # boundary的上采样

        self.contraction_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )
        self.expansion_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )
        self.boundary_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1,
                                         padding=0)  # 预测boundary的卷积核

        # extra mask supervision

        self.ce_module = ContractionExpansionModule(cfg, input_shape.channels)

        self.expansion_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)  # 收缩部分的预测器
        self.contraction_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)  # 膨胀部分的预测器

        for layer in self.mask_fcns + \
                     [self.mask_deconv, self.boundary_deconv, self.boundary_to_mask,
                      self.mask_final_fusion, self.downsample]:
            weight_init.c2_msra_fill(layer)  # 初始化参数

        # use normal distribution initialization for mask prediction layer          对于所有的预测器都使用正态分布初始化
        nn.init.normal_(self.mask_predictor.weight, std=0.001)
        nn.init.normal_(self.boundary_predictor.weight, std=0.001)
        nn.init.normal_(self.expansion_predictor.weight, std=0.001)
        nn.init.normal_(self.contraction_predictor.weight, std=0.001)

        if self.mask_predictor.bias is not None:
            nn.init.constant_(self.mask_predictor.bias, 0)
        if self.boundary_predictor.bias is not None:
            nn.init.constant_(self.boundary_predictor.bias, 0)
        if self.contraction_predictor.bias is not None:
            nn.init.constant_(self.contraction_predictor.bias, 0)
        if self.expansion_predictor.bias is not None:
            nn.init.constant_(self.expansion_predictor.bias, 0)

    def forward(self, mask_features, boundary_features, instances: List[Instances]):
        """
        Args:
            mask_features[torch.Tensor] mask features for all roi in all the images shape=[M, C, 14, 14], generally, the resolution of mask features is 14
            boundary_features[torch.Tensor] mask boundary features for all rois in all the images shape=[M, C, 28, 28], the resolution of mask boundary features is 28
            instances (list[Instances])
        """
        for layer in self.mask_fcns:
            mask_features = layer(mask_features)  # [M, C, 14, 14]
        if mask_features.shape[0] == 0:
            return instances
        # downsample
        boundary_features = self.downsample(boundary_features)  # [M, C, 14, 14]

        boundary_features, contraction_features, expansion_features = self.ce_module(mask_features, boundary_features)
        mask_features = mask_features + self.boundary_to_mask(
            boundary_features) + contraction_features + expansion_features

        mask_features = self.mask_final_fusion(mask_features)  # [M, C, 14, 14]
        # mask features
        mask_features = F.relu(self.mask_deconv(mask_features))  # [M, C, 28, 28]

        # mask prediction
        mask_logits = self.mask_predictor(mask_features)

        if self.training:
            # extra loss
            contraction_features = F.relu(self.contraction_deconv(contraction_features))
            contraction_logits = self.contraction_predictor(contraction_features)

            expansion_features = F.relu(self.expansion_deconv(expansion_features))
            expansion_logits = self.expansion_predictor(expansion_features)

            boundary_features = F.relu(self.boundary_deconv(boundary_features))
            boundary_logits = self.boundary_predictor(boundary_features)

            loss_mask, loss_boundary, loss_contraction, loss_expansion = joint_loss(mask_logits,
                                                                                    contraction_logits,
                                                                                    expansion_logits,
                                                                                    boundary_logits,
                                                                                    instances,
                                                                                    kernel_size=self.kernel_size)
            return {
                "loss_mask": loss_mask * self.loss_weight[0],
                "loss_boundary": loss_boundary * self.loss_weight[1],
                "loss_contraction": loss_contraction * self.loss_weight[2],
                "loss_expansion": loss_expansion * self.loss_weight[3]
            }
        else:
            mask_rcnn_inference(mask_logits, instances)
            return instances