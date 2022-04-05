import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
import numpy as np

from typing import List, Optional
from cvpods.layers import position_encoding_dict
from cvpods.modeling.backbone import Transformer
from cvpods.modeling.matcher import HungarianMatcher
from cvpods.modeling.postprocessing import detector_postprocess
from cvpods.structures import Boxes, ImageList, Instances, BitMasks
from cvpods.structures import boxes as box_ops
from cvpods.structures.boxes import generalized_box_iou
from cvpods.utils import comm
from cvpods.utils.metrics import accuracy

__all__ = ["DETRVIS"]


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, x):
        xs = self.body(x)
        return xs


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=FrozenBatchNorm2d)  # hack: currently, we do not use pretrained models from torch
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


def build_backbone(cfg):
    train_backbone = True
    return_interm_layers = cfg.MODEL.MASK_ON
    backbone = Backbone(cfg.MODEL.BACKBONE.NAME, train_backbone, return_interm_layers, cfg.MODEL.BACKBONE.DILATION)
    backbone = nn.Sequential(backbone) # hack for load pretrained models from detr
    return backbone


class DETRVIS(nn.Module):
    def __init__(self, cfg, freeze_detr=False):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        # Build Backbone
        self.backbone = build_backbone(cfg)

        # Build Transformer
        self.transformer = Transformer(cfg)

        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.num_queries = cfg.MODEL.DETR.NUM_QUERIES
        hidden_dim = self.transformer.d_model

        # Build FFN
        self.class_embed = nn.Linear(hidden_dim, self.num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # Build Object Queries
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        backbone_out_channels = self.backbone[0].num_channels
        self.input_proj = nn.Conv2d(backbone_out_channels, hidden_dim, kernel_size=1)

        self.position_embedding = position_encoding_dict[cfg.MODEL.DETR.POSITION_EMBEDDING](
            num_pos_feats=hidden_dim // 2,
            temperature=cfg.MODEL.DETR.TEMPERATURE,
            normalize=True if cfg.MODEL.DETR.POSITION_EMBEDDING == "sine" else False,
            scale=None,
        )

        self.match_coef = cfg.MODEL.DETR.TRACKING.MATCH_COEF

        self.weight_dict = {
            "loss_ce": cfg.MODEL.DETR.CLASS_LOSS_COEFF,
            "loss_bbox": cfg.MODEL.DETR.BBOX_LOSS_COEFF,
            "loss_giou": cfg.MODEL.DETR.GIOU_LOSS_COEFF,
            "loss_mask": cfg.MODEL.DETR.MASK_LOSS_COEFF,
            "loss_track": cfg.MODEL.DETR.TRACK_LOSS_COEFF
        }

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)


        self.aux_loss = not cfg.MODEL.DETR.NO_AUX_LOSS
        if self.aux_loss:
            self.aux_weight_dict = {}
            for i in range(cfg.MODEL.DETR.TRANSFORMER.NUM_DEC_LAYERS - 1):
                self.aux_weight_dict.update({k + f"_{i}": v for k, v in self.weight_dict.items()})
            self.weight_dict.update(self.aux_weight_dict)

        losses = ["labels", "boxes", "cardinality", "masks", "track"]

        matcher = HungarianMatcher(
            cost_class=cfg.MODEL.DETR.COST_CLASS,
            cost_bbox=cfg.MODEL.DETR.COST_BBOX,
            cost_giou=cfg.MODEL.DETR.COST_GIOU,
        )

        self.criterion = SetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=self.weight_dict,
            eos_coef=cfg.MODEL.DETR.EOS_COEFF,
            losses=losses,
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)

        if not cfg.MODEL.RESNETS.STRIDE_IN_1X1:
            # Custom or torch pretrain weights
            self.normalizer = lambda x: (x / 255.0 - pixel_mean) / pixel_std
        else:
            # MSRA pretrain weights
            self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)
        self.tracking_head = TrackHeadSmallConv(fpn_dims=[2048, 1024, 512], context_dim=256,
                                                out_dim=512, memory_dim=256)

        self.prev_roi_feats = None
        self.prev_bboxes = None
        self.prev_det_labels = None

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """

        images = self.preprocess_image(batched_inputs)

        B, C, H, W = images.tensor.shape
        device = images.tensor.device

        mask = torch.ones((B, H, W), dtype=torch.bool, device=device)
        for img_shape, m in zip(images.image_sizes, mask):
            m[: img_shape[0], : img_shape[1]] = False

        features = self.backbone(images.tensor)
        src = features["3"]
        bs = src.shape[0]
        mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).bool()[0]
        pos = self.position_embedding(src, mask)

        src_proj = self.input_proj(src)
        hs, memory = self.transformer(src_proj, mask, self.query_embed.weight, pos)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)

        seg_masks = self.mask_head(src_proj, bbox_mask,
                                   [features["2"], features["1"], features["0"]])
        outputs_seg_masks = seg_masks.view(bs, self.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        out["pred_masks"] = outputs_seg_masks

        tracking_features = self.tracking_head(memory, [features["3"], features["2"], features["1"]])
        out["tracking_features"] = tracking_features

        if self.training:
            reference_images = self.preprocess_reference_image(batched_inputs)

            reference_mask = torch.ones((B, H, W), dtype=torch.bool, device=device)
            for img_shape, m in zip(reference_images.image_sizes, reference_mask):
                m[: img_shape[0], : img_shape[1]] = False
            reference_features = self.backbone(reference_images.tensor)
            reference_src = reference_features["3"]
            reference_mask = F.interpolate(reference_mask[None].float(), size=reference_src.shape[-2:]).bool()[0]
            reference_src_proj = self.input_proj(reference_src)

            reference_pos = self.position_embedding(reference_src, reference_mask)
            reference_memory = self.transformer.forward_encoder(reference_src_proj, reference_mask, reference_pos)

            reference_tracking_features = self.tracking_head(reference_memory, [reference_features["3"],
                                                                                reference_features["2"],
                                                                                reference_features["1"]])
            out["reference_tracking_features"] = reference_tracking_features

            targets = self.convert_anno_format(batched_inputs)

            if self.aux_loss:
                out["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]

            loss_dict = self.criterion(out, targets)
            for k, v in loss_dict.items():
                loss_dict[k] = v * self.weight_dict[k] if k in self.weight_dict else v
            return loss_dict

        else:
            box_cls = out["pred_logits"]
            box_pred = out["pred_boxes"]
            is_first = batched_inputs[0]["is_first"]

            results = self.inference(box_cls, box_pred, outputs_seg_masks, tracking_features,
                                     images.image_sizes, is_first=is_first)

            processed_results = []

            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})

            return processed_results

        # FIXME h_boxes takes the last one computed, keep this in mind

    def inference(self, box_cls, box_pred, mask_pred, tracking_features, image_sizes, is_first):
        """
        Args:
            box_cls (Tensor): shape=[batch_size, num_queries, K].
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): shape=[batch_size, num_queries, 4].
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every query.
            mask_pred (Tensor): shape=[batch_size, num_queries, h, w], stride=8
                The tensor predicts the instance mask for each query.
            tracking_features(Tensor): shape=[batch_size, tracking_feature_dim, h, w], stride=8
                The tensor predicts the tracking features used for tracking.
            image_sizes(List(torch.Size)): The input image sizes
            is_first(Bool): Whether this image is the first frame of one video

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, tracking_feature_per_image, image_size) in \
                enumerate(zip(scores, labels, box_pred, tracking_features, image_sizes)):
            # Post process for detection
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image

            # Post process for segmentation
            mask = F.interpolate(mask_pred[i].unsqueeze(0), size=image_size, mode='bilinear', align_corners=False)
            mask = mask[0].sigmoid() > 0.5
            B, N, H, W = mask_pred.shape
            mask = BitMasks(mask.cpu()).crop_and_resize(result.pred_boxes.tensor.cpu(), 32)
            result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            # Post process for tracking
            det_roi_feats = self._extract_box_feature_center_single(tracking_feature_per_image,
                                                                    box_pred_per_image)
            if is_first or (not is_first and self.prev_bboxes is None):
                self.prev_roi_feats = det_roi_feats
                self.prev_bboxes = result.pred_boxes.tensor
                self.prev_det_labels = labels_per_image
                det_obj_ids = torch.arange(labels_per_image.size(0))
                result.pred_obj_ids = det_obj_ids
            else:
                assert self.prev_roi_feats is not None
                prod = torch.mm(det_roi_feats, torch.transpose(self.prev_roi_feats, 0, 1))
                pred_boxes = result.pred_boxes.tensor
                n = prod.size(0)
                dummy = torch.zeros(n, 1, device=torch.cuda.current_device())
                match_score = torch.cat([dummy, prod], dim=1)
                mat_logprob = F.log_softmax(match_score, dim=1)
                label_delta = (self.prev_det_labels == labels_per_image.view(-1, 1)).float()
                bbox_ious = box_ops.pairwise_iou_tensor(pred_boxes, self.prev_bboxes)
                comp_scores = self._compute_comp_scores(mat_logprob,
                                                        scores_per_image.view(-1, 1),
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
                        self.prev_roi_feats = torch.cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                        self.prev_bboxes = torch.cat((self.prev_bboxes, pred_boxes[idx][None]), dim=0)
                        self.prev_det_labels = torch.cat((self.prev_det_labels, labels_per_image[idx][None]), dim=0)
                    else:
                        obj_id = match_id - 1
                        match_score = comp_scores[idx, match_id]
                        if match_score > best_match_scores[obj_id]:
                            det_obj_ids[idx] = obj_id
                            best_match_scores[obj_id] = match_score
                            self.prev_roi_feats[obj_id] = det_roi_feats[idx]
                            self.prev_bboxes[obj_id] = pred_boxes[idx]
                result.pred_obj_ids = det_obj_ids
            results.append(result)
        return results

    def _extract_box_feature_center_single(self, track_feats, boxes):
        """
        Args:
            track_feats[torch.Tensor]: tracking features of one image stride=8, shape=[512, h, w]
            boxes[torch.Tensor]: predicted boxes(query image) or gt boxes(reference image)
             [0-1](cx, cy, w, h) shape=[num_boxes, 4]
        """
        track_box_feats = track_feats.new_zeros(boxes.size(0), track_feats.shape[0])
        h, w = track_feats.shape[-2:]
        boxes_center_xs = torch.floor(boxes[:, 0] * w).long()
        boxes_center_ys = torch.floor(boxes[:, 1] * h).long()

        aa = track_feats.permute(1, 2, 0)
        bb = aa[boxes_center_ys, boxes_center_xs, :]
        track_box_feats += bb
        return track_box_feats

    def _compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy):
        if add_bbox_dummy:
            dummy_iou = torch.ones(bbox_ious.size(0), 1, device=torch.cuda.current_device()) * 0
            bbox_ious = torch.cat([dummy_iou, bbox_ious], dim=1)
            dummy_label = torch.ones(bbox_ious.size(0), 1, device=torch.cuda.current_device())
            label_delta = torch.cat([dummy_label, label_delta], dim=1)
        if self.match_coef is None:
            return match_ll
        else:
            assert len(self.match_coef) == 3
            return match_ll + self.match_coef[0] * torch.log(bbox_scores) + \
                   self.match_coef[1] * bbox_ious + self.match_coef[2] * label_delta

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].float().to(self.device) for x in batched_inputs]
        images = [self.normalizer(img) for img in images]
        images = ImageList.from_tensors(images, 32)
        return images

    def preprocess_reference_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image_reference"].float().to(self.device) for x in batched_inputs]
        images = [self.normalizer(img) for img in images]
        images = ImageList.from_tensors(images, 32)
        return images

    def convert_anno_format(self, batched_inputs):
        targets = []
        for bi in batched_inputs:
            target = {}
            h, w = bi["image"].shape[-2:]
            boxes = box_ops.box_xyxy_to_cxcywh(
                bi["instances"].gt_boxes.tensor / torch.tensor([w, h, w, h], dtype=torch.float32)
            )
            target["boxes"] = boxes.to(self.device)
            if "instances_reference" in bi:
                reference_boxes = box_ops.box_xyxy_to_cxcywh(
                    bi["instances_reference"].gt_boxes.tensor / torch.tensor([w, h, w, h], dtype=torch.float32)
                )
                target["reference_boxes"] = reference_boxes
            target["area"] = bi["instances"].gt_boxes.area().to(self.device)
            target["labels"] = bi["instances"].gt_classes.to(self.device)
            target["masks"] = bi["instances"].gt_masks.to(self.device)
            target["gt_pids"] = bi["instances"].gt_pids.to(self.device)
            if hasattr(bi["instances"], "gt_masks"):
                target["masks"] = bi["instances"].gt_masks
            target["iscrowd"] = torch.zeros_like(target["labels"], device=self.device)
            target["orig_size"] = torch.tensor([bi["height"], bi["width"]], device=self.device)
            target["size"] = torch.tensor([h, w], device=self.device)
            targets.append(target)

        return targets


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class TrackHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, fpn_dims, context_dim, out_dim, memory_dim):
        super(TrackHeadSmallConv, self).__init__()

        self.adapter1 = nn.Conv2d(fpn_dims[0], context_dim, 1)
        self.adapter2 = nn.Conv2d(fpn_dims[1], context_dim, 1)
        self.adapter3 = nn.Conv2d(fpn_dims[2], context_dim, 1)

        self.lay1 = nn.Conv2d(memory_dim, context_dim, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(32, context_dim)
        self.lay2 = nn.Conv2d(context_dim, context_dim, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(32, context_dim)
        self.lay3 = nn.Conv2d(context_dim, context_dim, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(32, context_dim)
        self.out_lay = nn.Conv2d(context_dim, out_dim, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, memory, fpns):
        """
        Args:
            memory[torch.Tensor]: Output of encoder shape=[bs, 256, H, W] stride 32
            fpns [list(torch.Tensor)] shape_i = [bs, C, H, W] layer1, layer2, layer3 stride 4 8 16
        """
        cur_fpn = self.adapter1(fpns[0])                    # [bs, 256, H, W]   stride=32
        x = cur_fpn + memory                                # [bs, 256, H, W] stride=32
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])                    # [bs, 256, H, W]   stride=16
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="bilinear", align_corners=True)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])                    # [bs, 256, H, W]   stride=8
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="bilinear", align_corners=True)
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        x = self.out_lay(x)

        return x                                            # [bs, 512, H, W]    stride 8


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, bbox_mask: torch.Tensor, fpns: List[torch.Tensor]):
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[torch.Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights

def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _extract_box_feature_center_single(self, track_feats, boxes):
        """
        Args:
            track_feats[torch.Tensor]: tracking features of one image stride=8, shape=[512, h, w]
            boxes[torch.Tensor]: predicted boxes(query image) or gt boxes(reference image)
             [0-1](cx, cy, w, h) shape=[num_boxes, 4]
        """
        track_box_feats = track_feats.new_zeros(boxes.size(0), track_feats.shape[0])
        h, w = track_feats.shape[-2:]
        boxes_center_xs = torch.floor(boxes[:, 0] * w).long()
        boxes_center_ys = torch.floor(boxes[:, 1] * h).long()

        aa = track_feats.permute(1, 2, 0)
        bb = aa[boxes_center_ys, boxes_center_xs, :]
        track_box_feats += bb
        return track_box_feats

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        del num_boxes

        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, ie the absolute error in the number of predicted non-empty
        boxes. This is not really a loss, it is intended for logging purposes only. It doesn't
        propagate gradients
        """
        del indices
        del num_boxes
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the
        image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        def helper(tensor_list: List[torch.Tensor]):
            assert tensor_list[0].ndim == 3
            max_size = list(max(s) for s in zip(*[tensor.shape for tensor in tensor_list]))
            batch_shape = [len(tensor_list)] + max_size
            b, c, h, w = batch_shape
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
                m[:img.shape[1], :img.shape[2]] = False
            return tensor, mask

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"].tensor for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = helper(masks)
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                  mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_track(self, outputs, targets, indices, num_boxes):
        """Conpute the losses related to the tracking head
            Args:
                targets: list of dict with keys boxes, labels etc
        """
        assert "tracking_features" in outputs
        assert "reference_tracking_features" in outputs
        query_tracking_features = outputs["tracking_features"]                      # [bs, 512, h, w] stride=8
        reference_tracking_features = outputs["reference_tracking_features"]        # [bs, 512, h, w] stride=8
        loss_match = 0
        n_total = 0

        # 1. Getting training gt pids using gt pids and indices[1], each element with value from 0 to num_reference_boxes
        gt_pids = [t['gt_pids'][i] for t, (_, i) in zip(targets, indices)]          # list of tensor len_i=num_gt_boxes_this_query_image
        # 2. Getting predicted query boxes using all pred_boxes and indices[0]
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]                                      # [num_query_gts_whole_batch, 4]
        num_boxes_per_image = [len(src) for (src, _) in indices]                    # [batch_size, ]
        src_boxes = torch.split(src_boxes, num_boxes_per_image)                     # list of tensors shape_i=[num_query_gts_ith_image, 4]
        # 3. Extracting features from query tracking features and reference tracking features and calculate prod
        reference_boxes = [token["reference_boxes"] for token in targets]           # list of tensors shape_i=[num_reference_boxes, 4]
        prods = []
        for reference_box, src_box, reference_feature, query_feature in \
                zip(reference_boxes, src_boxes, reference_tracking_features, query_tracking_features): #Loop for each image
            reference_box_features = self._extract_box_feature_center_single(reference_feature, reference_box)      # [num_gt_reference_boxes_this_image, 512]
            query_box_features = self._extract_box_feature_center_single(query_feature, src_box)        # [num_gt_query_boxes_this_image, 512]
            prod = torch.mm(query_box_features, torch.transpose(reference_box_features, 0, 1))
            n = prod.size(0)
            dummy = torch.zeros(n, 1, device=torch.cuda.current_device())
            prod_ext = torch.cat([dummy, prod], dim=1)                              # [num_gt_query_box, num_gt_reference_box+1]
            prods.append(prod_ext)
        # 4. Calculating match loss
        num_images = query_tracking_features.shape[0]
        for prod, pids in zip(prods, gt_pids):
            if prod.numel() == 0:
                loss_match += prod.sum() * 0
                continue
            loss_match += F.cross_entropy(prod, pids, reduction='mean')
            n_total += len(pids)
        loss_match = loss_match / num_images
        losses = {"loss_match": loss_match}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'track': self.loss_track
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        if comm.get_world_size() > 1:
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / comm.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if loss == 'track':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x