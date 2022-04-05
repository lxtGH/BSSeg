import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np

from cvpods.layers import cat, ShapeSpec, conv_with_kaiming_uniform, roi_align
from cvpods.structures import BitMasks, polygons_to_bitmask
from cvpods.layers.mask_ops import BYTES_PER_FLOAT, GPU_MEM_LIMIT, _do_paste_mask
from ..poolers import ROIPooler, convert_boxes_to_pooler_format


def generate_block_target(mask_target, boundary_width=3):
    """
    For the first refinement resule
    """
    mask_target = mask_target.float()  # (n, 28, 28)

    # boundary region
    kernel_size = 2 * boundary_width + 1  # 2*2+1 = 5
    laplacian_kernel = - torch.ones(1, 1, kernel_size, kernel_size).to(
        dtype=torch.float32, device=mask_target.device).requires_grad_(False)  # all elements equals to -1
    laplacian_kernel[0, 0, boundary_width, boundary_width] = kernel_size ** 2 - 1  # 24

    pad_target = F.pad(mask_target.unsqueeze(1), (boundary_width, boundary_width, boundary_width, boundary_width),
                       "constant", 0)

    # pos_boundary
    pos_boundary_targets = F.conv2d(pad_target, laplacian_kernel, padding=0)
    pos_boundary_targets = pos_boundary_targets.clamp(min=0) / float(kernel_size ** 2)
    pos_boundary_targets[pos_boundary_targets > 0.1] = 1
    pos_boundary_targets[pos_boundary_targets <= 0.1] = 0
    pos_boundary_targets = pos_boundary_targets.squeeze(1)

    # neg_boundary
    neg_boundary_targets = F.conv2d(1 - pad_target, laplacian_kernel, padding=0)
    neg_boundary_targets = neg_boundary_targets.clamp(min=0) / float(kernel_size ** 2)
    neg_boundary_targets[neg_boundary_targets > 0.1] = 1
    neg_boundary_targets[neg_boundary_targets <= 0.1] = 0
    neg_boundary_targets = neg_boundary_targets.squeeze(1)

    # generate block target
    block_target = torch.zeros_like(mask_target).long().requires_grad_(False)
    boundary_inds = (pos_boundary_targets + neg_boundary_targets) > 0
    foreground_inds = (mask_target - pos_boundary_targets) > 0
    block_target[boundary_inds] = 1
    block_target[foreground_inds] = 2
    return block_target


class RefineCrossEntropyLoss(torch.nn.Module):

    def __init__(self, cfg):
        super(RefineCrossEntropyLoss, self).__init__()

        self.stage_instance_loss_weight = cfg.MODEL.REFINE_MASK_HEAD.LOSS.STAGE_INSTANCE_LOSS_WEIGHT
        # mask loss weight for the first prediction and another three refinement
        self.semantic_loss_weight = cfg.MODEL.REFINE_MASK_HEAD.LOSS.SEMANTIC_LOSS_WEIGHT
        self.boundary_width = cfg.MODEL.REFINE_MASK_HEAD.LOSS.BOUNDARY_WIDTH
        self.start_stage = cfg.MODEL.REFINE_MASK_HEAD.LOSS.START_STAGE

    def forward(self, stage_instance_preds, semantic_pred, stage_instance_targets, semantic_target):
        """
        Args:
            stage_instance_preds (list of Tensor), instance predictions for the initial prediction and three
                refinement predictions, shape_i=(num_instances, 1, 14/28/56/112, 14, 28, 56, 112)
            semantic_pred (Tensor), semantic predictions shape=(bs, 1, h, w) stride4
            stage_instance_targets (list of Tensor), A list of N Tensor, where N means the number of stages, instance
            targets for the initial prediction and three consecutive refinement stages
            shape=[num_instances, 14/28/56/112, 14/28/56/112]
            semantic_target (Tensor), Ground truth of the semantic prediction, shape=[bs, h, w], shape=stride4
        """
        loss_mask_set = []
        for idx in range(len(stage_instance_preds)):  # Loop for each stage's result
            instance_pred, instance_target = stage_instance_preds[idx].squeeze(1), stage_instance_targets[
                idx]                        # (num_instances, 14/28/56/112, 14/28/56/112)
            if idx <= self.start_stage:  # init pred and the first refinement
                loss_mask = F.binary_cross_entropy_with_logits(
                    instance_pred, instance_target.to(dtype=torch.float32), reduction="mean")  # instance loss
                loss_mask_set.append(loss_mask)
                pre_pred = instance_pred.sigmoid() >= 0.5  # initial mask or the first refinement
            else:                        # for the latter two boundary aware refinements
                pre_boundary = generate_block_target(pre_pred.float(),
                                                     boundary_width=self.boundary_width) == 1  # boundary of the first/second predict refinement mask, boundary part equals to 1
                boundary_region = pre_boundary.unsqueeze(1)  # (n, 1, 28/56, 28/56)

                target_boundary = generate_block_target(
                    stage_instance_targets[idx - 1].float(),
                    boundary_width=self.boundary_width) == 1  # boundary of the first/second ground truth refinement mask, boundary part equals to 1
                boundary_region = boundary_region | target_boundary.unsqueeze(1)  # (n, 1, 28/56, 28/56)

                boundary_region = F.interpolate(
                    boundary_region.float(),
                    instance_pred.shape[-2:], mode='bilinear', align_corners=True)  # 28/56 -> 56/112
                boundary_region = (boundary_region >= 0.5).squeeze(1)

                loss_mask = F.binary_cross_entropy_with_logits(instance_pred, instance_target, reduction='none')
                loss_mask = loss_mask[boundary_region].sum() / boundary_region.sum().clamp(min=1).float()
                loss_mask_set.append(
                    loss_mask)  # This loss function means in the latter two refinement, model only focuses on the boundary

                # generate real mask pred, set boundary width as 1, same as inference
                pre_boundary = generate_block_target(pre_pred.float(), boundary_width=1) == 1  # (n, 1, 28/56, 28/56)

                pre_boundary = F.interpolate(
                    pre_boundary.unsqueeze(1).float(),
                    instance_pred.shape[-2:], mode='bilinear', align_corners=True) >= 0.5       # 28/56 ->56/112

                pre_pred = F.interpolate(
                    stage_instance_preds[idx - 1],
                    instance_pred.shape[-2:], mode='bilinear', align_corners=True)      # 28/56 ->56/112

                pre_pred[pre_boundary] = stage_instance_preds[idx][
                    pre_boundary]  # Using the boundary prediction of the last two refinement replace the first two boundary prediction
                pre_pred = pre_pred.squeeze(1).sigmoid() >= 0.5             # 56/112

        assert len(self.stage_instance_loss_weight) == len(loss_mask_set)
        loss_instance = sum([weight * loss for weight, loss in zip(self.stage_instance_loss_weight, loss_mask_set)])
        loss_semantic = self.semantic_loss_weight * \
                        F.binary_cross_entropy_with_logits(semantic_pred.squeeze(1), semantic_target)

        return loss_instance, loss_semantic


class MultiBranchFusion(nn.Module):

    def __init__(self, feat_dim, dilations=[1, 3, 5]):
        super(MultiBranchFusion, self).__init__()

        conv_block = conv_with_kaiming_uniform(norm=None, activation=True)

        for idx, dilation in enumerate(dilations):
            self.add_module(f'dilation_conv_{idx + 1}', conv_block(
                feat_dim, feat_dim, kernel_size=3, stride=1, padding=dilation, dilation=dilation))

        self.merge_conv = conv_block(feat_dim, feat_dim, kernel_size=1, stride=1, activation=None)

    def forward(self, x):
        feat_1 = self.dilation_conv_1(x)
        feat_2 = self.dilation_conv_2(x)
        feat_3 = self.dilation_conv_3(x)
        out_feat = self.merge_conv(feat_1 + feat_2 + feat_3)
        return out_feat


class SFMStage(nn.Module):

    def __init__(self,
                 semantic_in_channel=256,
                 semantic_out_channel=256,
                 instance_in_channel=256,
                 instance_out_channel=256,
                 dilations=[1, 3, 5],
                 out_size=14,
                 num_classes=80,
                 semantic_out_stride=4,
                 mask_use_sigmoid=False,
                 upsample='bilinear'):
        super(SFMStage, self).__init__()

        self.semantic_out_stride = semantic_out_stride
        self.mask_use_sigmoid = mask_use_sigmoid
        self.num_classes = num_classes

        # for extracting instance-wise semantic feats
        self.semantic_transform_in = nn.Conv2d(semantic_in_channel, semantic_out_channel, 1)
        self.semantic_roi_extractor = ROIPooler(output_size=out_size,
                                                scales=[1.0 / semantic_out_stride, ],
                                                sampling_ratio=0,
                                                pooler_type="ROIAlignV2")
        self.semantic_transform_out = nn.Conv2d(semantic_out_channel, semantic_out_channel, 1)

        self.instance_logits = nn.Conv2d(instance_in_channel, num_classes, 1)           # predict instance results

        fuse_in_channel = instance_in_channel + semantic_out_channel + 2                # +2 for instance prediction and semantic prediction
        self.fuse_conv = nn.ModuleList([
            nn.Conv2d(fuse_in_channel, instance_in_channel, 1),
            MultiBranchFusion(instance_in_channel, dilations=dilations)])               # some parallel convs

        self.fuse_transform_out = nn.Conv2d(instance_in_channel, instance_out_channel - 2, 1)           # -2 for add instance and semantic predictions
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample, align_corners=False)
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in [self.semantic_transform_in, self.semantic_transform_out, self.instance_logits, self.fuse_transform_out]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in self.fuse_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, semantic_feats, semantic_pred, rois, roi_labels):
        """
        Args:
            instance_feats: [num_instance, 256, 14, 14] torch.Tensor
            semantic_feats: [bs, 256, h, w]    torch.Tensor stride=4
            semantic_pred: [bs, 1, h, w]      torch.Tensor stride=4
            rois (list of Tensor): len=batch size, shape=(num_instances_each_image, 4)
            roi_labels: (Tensor): shape=(num_instances_whole_batch, )
        """
        concat_tensors = [instance_feats]

        # instance-wise semantic feats
        semantic_feat = self.relu(self.semantic_transform_in(semantic_feats))            # [bs, 256, h, w] stride4
        ins_semantic_feats = self.semantic_roi_extractor([semantic_feat,], rois)        # [num_instances, 256, 14, 14]
        ins_semantic_feats = self.relu(self.semantic_transform_out(ins_semantic_feats))  # [num_instances, 256, 14, 14]
        concat_tensors.append(ins_semantic_feats)               # instance features + semantic features (256+256)

        # instance masks
        instance_preds = self.instance_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]     #  [num_instance, 1, 14, 14]
        _instance_preds = instance_preds.sigmoid() if self.mask_use_sigmoid else instance_preds
        instance_masks = F.interpolate(_instance_preds, instance_feats.shape[-2], mode='bilinear', align_corners=True)
        concat_tensors.append(instance_masks)                   # instance features + semantic features + instance predictions (256+256+1)

        # instance-wise semantic masks
        fake_rois = rois.clone()
        fake_rois = convert_boxes_to_pooler_format(fake_rois)   # (num_instances, 5)
        _semantic_pred = semantic_pred.sigmoid() if self.mask_use_sigmoid else semantic_pred        # (bs, 1, h, w) stride 4
        ins_semantic_masks = roi_align(
            _semantic_pred, fake_rois, instance_feats.shape[-2:], 1.0 / self.semantic_out_stride, 0, True)  # (num_instances, 1, 14, 14)
        ins_semantic_masks = F.interpolate(
            ins_semantic_masks, instance_feats.shape[-2:], mode='bilinear', align_corners=True)             # (num_instances, 1, 14, 14)
        concat_tensors.append(ins_semantic_masks)   # instance features + semantic features + instance predictions + semantic predictions (256+256+1+1)

        # fuse instance feats & instance masks & semantic feats & semantic masks
        fused_feats = cat(concat_tensors, dim=1)                  # [num_instances, 256+256+1+1, 14, 14]
        for conv in self.fuse_conv:
            fused_feats = self.relu(conv(fused_feats))                  # (num_instances, 256, 14, 14)

        fused_feats = self.relu(self.fuse_transform_out(fused_feats))       # (num_instances, 126, 14, 14)
        fused_feats = self.relu(self.upsample(fused_feats))                 # (num_instances, 124, 28, 28)

        # concat instance and semantic masks with fused feats again
        instance_masks = F.interpolate(_instance_preds, fused_feats.shape[-2], mode='bilinear', align_corners=True)             # (num_instances, 1, 28, 28)
        ins_semantic_masks = F.interpolate(ins_semantic_masks, fused_feats.shape[-2], mode='bilinear', align_corners=True)      # (num_instances, 1, 28, 28)
        fused_feats = cat([fused_feats, instance_masks, ins_semantic_masks], dim=1)     # (num_instances, 128, 28, 28)

        return instance_preds, fused_feats


class RefineMaskHead(nn.Module):

    def __init__(self, cfg, input_shape: ShapeSpec):
        super(RefineMaskHead, self).__init__()

        self.num_convs_instance = cfg.MODEL.REFINE_MASK_HEAD.NUM_INSTANCE_CONVS
        self.conv_kernel_size_instance = cfg.MODEL.REFINE_MASK_HEAD.INSTANCE_KERNEL_SIZE
        self.conv_in_channels_instance = cfg.MODEL.REFINE_MASK_HEAD.INSTANCE_IN_CHANNELS
        self.conv_out_channels_instance = cfg.MODEL.REFINE_MASK_HEAD.INSTANCE_OUT_CHANNELS

        self.num_convs_semantic = cfg.MODEL.REFINE_MASK_HEAD.NUM_SEMANTIC_CONVS
        self.conv_kernel_size_semantic = cfg.MODEL.REFINE_MASK_HEAD.SEMANTIC_KERNEL_SIZE
        self.conv_in_channels_semantic = cfg.MODEL.REFINE_MASK_HEAD.SEMANTIC_IN_CHANNELS
        self.conv_out_channels_semantic = cfg.MODEL.REFINE_MASK_HEAD.SEMANTIC_OUT_CHANNELS

        self.norm_cfg = cfg.MODEL.REFINE_MASK_HEAD.NORM

        self.semantic_out_stride = cfg.MODEL.REFINE_MASK_HEAD.SEMANTIC_OUT_STRIDE
        self.stage_sup_size = cfg.MODEL.REFINE_MASK_HEAD.STAGE_SUP_SIZE          # [14, 28, 56, 128]
        self.stage_num_classes = cfg.MODEL.REFINE_MASK_HEAD.STAGE_NUM_CLASSES

        self._build_conv_layer('instance')                      # (3x3 conv + relu)x2
        self._build_conv_layer('semantic')                      # (3x3 conv + relu)x4
        self.loss_func = RefineCrossEntropyLoss(cfg)

        assert len(self.stage_sup_size) > 1
        self.stages = nn.ModuleList()
        out_channel = cfg.MODEL.REFINE_MASK_HEAD.CONV_OUT_CHANNELS_INSTANCE                # 256
        for idx, out_size in enumerate(self.stage_sup_size[:-1]):           # 14, 28, 56
            in_channel = out_channel
            out_channel = in_channel // 2                           # before up sample, the fuse features in SFm are compressed with a 1x1 convlution layer to halve its channels.

            new_stage = SFMStage(
                semantic_in_channel=self.conv_out_channels_semantic,
                semantic_out_channel=in_channel,
                instance_in_channel=in_channel,
                instance_out_channel=out_channel,
                dilations=cfg.MODEL.REFINE_MASK_HEAD.SFM.DILATION,
                out_size=out_size,
                num_classes=self.stage_num_classes[idx],
                semantic_out_stride=self.semantic_out_stride,
                mask_use_sigmoid=cfg.MODEL.REFINE_MASK_HEAD.SFM.MASK_USE_SIGMOID,
                upsample=cfg.MODEL.REFINE_MASK_HEAD.SFM.UPSAMPLE)

            self.stages.append(new_stage)

        self.final_instance_logits = nn.Conv2d(out_channel, self.stage_num_classes[-1], 1)          # prediction for the final stage
        self.semantic_logits = nn.Conv2d(self.conv_out_channels_semantic, 1, 1)                          # semantic branch prediction
        self.relu = nn.ReLU(inplace=True)

    def _build_conv_layer(self, name):
        out_channels = getattr(self, f'conv_out_channels_{name}')
        conv_kernel_size = getattr(self, f'conv_kernel_size_{name}')

        convs = []
        conv_block = conv_with_kaiming_uniform(norm=None, activation=True)
        for i in range(getattr(self, f'num_convs_{name}')):
            in_channels = getattr(self, f'conv_in_channels_{name}') if i == 0 else out_channels
            conv = conv_block(in_channels, out_channels, kernel_size=conv_kernel_size,
                              stride=1, dilation=1, padding=1)
            convs.append(conv)

        self.add_module(f'{name}_convs', nn.ModuleList(convs))

    def init_weights(self):
        for m in [self.final_instance_logits, self.semantic_logits]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, semantic_feat, rois, roi_labels):
        """
        Args:
            instance_feats(torch.Tensor)  shape = (num_instances, 256, 14, 14)
            semantic_feat (torch.Tensor) shape = (n, c, h, w) stride=4 where n means number of images per mini-batch
            rois (list(torch.Tensor)) len=number images shape_i = (num_boxes_i, 4)
            roi_labels (list(torch.Tensor)) len=number images shape_i = (num_boxes, )
        Returns:
            stage_instance_preds (list of Tensor), instance predictions for the initial prediction and three
            refinement predictions, shape_i=(num_instances, 1, 14/28/56/112, 14, 28, 56, 112)
            semantic_pred (Tensor), semantic predictions, shape=(bs, 1, h, w) stride=4
        """
        for conv in self.instance_convs:                        # 2 consecutive convolution layers
            instance_feats = conv(instance_feats)               # [num_instance, 256, 14, 14]

        for conv in self.semantic_convs:                        # 4 consecutive convolution layers
            semantic_feat = conv(semantic_feat)                 # [n, 256, h, w]    stride=4

        semantic_pred = self.semantic_logits(semantic_feat)             # [n, 1, h, w]      stride=4

        stage_instance_preds = []
        for stage in self.stages:
            instance_preds, instance_feats = stage(instance_feats, semantic_feat, semantic_pred, rois, roi_labels)
            # (num_instances, 1, 14/28/56, 14/28/56), (num_instances, 128, 28/56/112, 28/56/112)
            stage_instance_preds.append(instance_preds)

        # for LVIS, use class-agnostic classifier for the last stage
        roi_labels_concat = cat(roi_labels, dim=0)          # (num_instances_whole_batch, )
        if self.stage_num_classes[-1] == 1:
            roi_labels_concat = roi_labels_concat.clamp(max=0)

        instance_preds = self.final_instance_logits(instance_feats)[torch.arange(len(rois)), roi_labels_concat][:, None]
        # (num_instances, 1, 112, 112)
        stage_instance_preds.append(instance_preds)

        return stage_instance_preds, semantic_pred

    def loss(self, stage_instance_preds, semantic_pred, stage_instance_targets, semantic_target):

        loss_instance, loss_semantic = self.loss_func(
            stage_instance_preds, semantic_pred, stage_instance_targets, semantic_target)

        return loss_instance, loss_instance

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale):

        mask_pred = mask_pred.sigmoid()

        device = mask_pred[0].device
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <= N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        if mask_pred.shape[1] > 1:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        im_segms = [im_mask[i].cpu().numpy() for i in range(N)]
        return im_segms
