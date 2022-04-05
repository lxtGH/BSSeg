import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from cvpods.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from cvpods.modeling.nn_utils import weight_init
from cvpods.utils import get_event_storage

class MaskIoUHead(nn.Module):

    def __init__(self, cfg, input_shape):
        super(MaskIoUHead, self).__init__()

        self.in_channels = input_shape.channels
        self.conv_out_channels = cfg.MODEL.ROI_MASK_HEAD.MASKIOUHEAD.CONV_OUT_CHANNELS
        self.fc_out_channels = cfg.MODEL.ROI_MASK_HEAD.MASKIOUHEAD.FC_OUT_CHANNELS
        self.num_classes = cfg.MODEL.ROI_MASK_HEAD.MASKIOUHEAD.NUM_CLASSES
        self.num_convs = cfg.MODEL.ROI_MASK_HEAD.MASKIOUHEAD.NUM_CONVS
        self.num_fcs = cfg.MODEL.ROI_MASK_HEAD.MASKIOUHEAD.NUM_FCS
        self.roi_feat_size = cfg.MODEL.ROI_MASK_HEAD.MASKIOUHEAD.ROI_FEAT_SIZE
        self.mask_iou_loss_weight = cfg.MODEL.ROI_MASK_HEAD.MASKIOUHEAD.LOSS_WEIGHT
        self.mask_iou_loss = nn.MSELoss()

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            if i == 0:
                in_channels = self.in_channels + 1
            else:
                in_channels = self.conv_out_channels
            stride = 2 if i == self.num_convs - 1 else 1
            self.convs.append(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=self.conv_out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1))

        roi_feat_size = _pair(self.roi_feat_size)
        pooled_area = (roi_feat_size[0] // 2) * (roi_feat_size[1] // 2)
        self.fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            in_channels = self.conv_out_channels * pooled_area if i == 0 else self.fc_out_channels
            self.fcs.append(nn.Linear(in_channels, self.fc_out_channels))

        self.fc_mask_iou = nn.Linear(self.fc_out_channels, self.num_classes)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        for conv in self.convs:
            weight_init.kaiming_init(conv)
        for fc in self.fcs:
            weight_init.kaiming_init(fc, a=1, mode='fan_in',
                                     nonlinearity='leaky_relu',
                                     distribution='uniform')
        nn.init.normal_(self.fc_mask_iou.weight, std=0.01)

    def forward(self, mask_feat, mask_pred):
        self.mask_side_len = mask_pred.size(2)
        if self.training:
            mask_pred = mask_pred.sigmoid()
        mask_pred_pooled = self.max_pool(mask_pred.unsqueeze(1))

        x = cat((mask_feat, mask_pred_pooled), dim=1)

        for conv in self.convs:
            x = self.relu(conv(x))
        x = x.flatten(1)
        for fc in self.fcs:
            x = self.relu(fc(x))

        mask_iou = self.fc_mask_iou(x)
        return mask_iou

    def get_target(self, mask_pred, instances, mask_targets):

        # calculate area ratios using full image shape mask
        area_ratios_all = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            area_ratios = []
            gt_masks = instances_per_image.gt_masks                 # image size
            proposals = instances_per_image.proposal_boxes.tensor
            num_pos = len(instances_per_image)
            for i in range(num_pos):
                gt_mask = gt_masks[i]
                proposal = proposals[i]
                gt_mask_in_proposal = gt_mask.crop(proposal)
                ratio = gt_mask_in_proposal.area[0] / (gt_mask.area[0] + 1e-7)
                area_ratios.append(ratio)
                area_ratios = torch.from_numpy(np.stack(area_ratios)).float().to(proposal.device)
            area_ratios_all.append(area_ratios)
        area_ratios = cat(area_ratios_all)


        mask_pred = (mask_pred > 0.5).float()                       # 28x28
        mask_pred_areas = mask_pred.sum((-1, -2))
        overlap_areas = (mask_pred * mask_targets).sum((-1, -2))

        gt_full_areas = mask_targets.sum((-1, -2)) / (area_ratios + 1e-7)

        mask_iou_targets = overlap_areas / (
            mask_pred_areas + gt_full_areas - overlap_areas)
        return mask_iou_targets


    def loss(self, mask_iou_pred, mask_iou_targets, instances):
        gt_classes = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)
        gt_classes = cat(gt_classes, dim=0)
        indices = torch.arange(mask_iou_pred.shape[0])
        mask_iou_pred = mask_iou_pred[indices, gt_classes]
        pos_inds = mask_iou_targets > 0
        if pos_inds.sum() > 0:
            loss_mask_iou = self.mask_iou_loss(mask_iou_pred[pos_inds],
                                               mask_iou_targets[pos_inds])
            loss_mask_iou = loss_mask_iou * self.mask_iou_loss_weight
        else:
            loss_mask_iou = mask_iou_pred.sum() * 0
        return dict(loss_mask_iou=loss_mask_iou)
