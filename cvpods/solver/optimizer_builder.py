#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List, Set
import itertools

import torch
from torch import optim

from cvpods.utils.registry import Registry

OPTIMIZER_BUILDER = Registry("Optimizer builder")

NORM_MODULE_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    # NaiveSyncBatchNorm inherits from BatchNorm2d
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
)


@OPTIMIZER_BUILDER.register()
class OptimizerBuilder:

    @staticmethod
    def build(model, cfg):
        raise NotImplementedError


@OPTIMIZER_BUILDER.register()
class SGDBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module in model.modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.OPTIMIZER.BASE_LR
                weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
                if isinstance(module, NORM_MODULE_TYPES):
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_NORM
                elif key == "bias":
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = cfg.SOLVER.OPTIMIZER.BASE_LR * cfg.SOLVER.OPTIMIZER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        optimizer = optim.SGD(
            params,
            cfg.SOLVER.OPTIMIZER.BASE_LR,
            momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM
        )
        return optimizer


@OPTIMIZER_BUILDER.register()
class AdamBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        lr = cfg.SOLVER.OPTIMIZER.BASE_LR
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY,
            amsgrad=cfg.SOLVER.OPTIMIZER.AMSGRAD
        )
        return optimizer


@OPTIMIZER_BUILDER.register()
class AdamWBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        lr = cfg.SOLVER.OPTIMIZER.BASE_LR
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=cfg.SOLVER.OPTIMIZER.BETAS,
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY,
            amsgrad=cfg.SOLVER.OPTIMIZER.AMSGRAD
        )
        return optimizer


@OPTIMIZER_BUILDER.register()
class SGDGateLRBuilder(OptimizerBuilder):
    """
    SGD Gate LR optimizer builder, used for DynamicRouting in cvpods.
    This optimizer will ultiply lr for gating function.
    """

    @staticmethod
    def build(model, cfg):
        gate_lr_multi = cfg.SOLVER.OPTIMIZER.GATE_LR_MULTI
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for name, module in model.named_modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.OPTIMIZER.BASE_LR
                weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
                if isinstance(module, NORM_MODULE_TYPES):
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_NORM
                elif key == "bias":
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = cfg.SOLVER.OPTIMIZER.BASE_LR * cfg.SOLVER.OPTIMIZER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY

                if gate_lr_multi > 0.0 and "gate_conv" in name:
                    lr *= gate_lr_multi

                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        optimizer = torch.optim.SGD(
            params,
            cfg.SOLVER.OPTIMIZER.BASE_LR,
            momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM
        )
        return optimizer

@OPTIMIZER_BUILDER.register()
class DETRAdamWBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        lr = cfg.SOLVER.OPTIMIZER.BASE_LR

        param_dicts = [
            {
                "params": [
                    param for name, param in model.named_parameters()
                    if "backbone" not in name and param.requires_grad]
            },
            {
                "params": [
                    param for name, param in model.named_parameters()
                    if "backbone" in name and param.requires_grad],
                "lr": cfg.SOLVER.OPTIMIZER.BASE_LR_RATIO_BACKBONE * lr,
            },
        ]

        optimizer = optim.AdamW(
            param_dicts,
            lr=lr,
            betas=cfg.SOLVER.OPTIMIZER.BETAS,
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY,
            amsgrad=cfg.SOLVER.OPTIMIZER.AMSGRAD
        )
        return optimizer


@OPTIMIZER_BUILDER.register()
class FullModelAdamWBuilder(OptimizerBuilder):

    @staticmethod
    def build(model, cfg):
        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        lr = cfg.SOLVER.OPTIMIZER.BASE_LR

        param_dicts = [
            {
                "params": [
                    param for name, param in model.named_parameters()
                    if "backbone" not in name and param.requires_grad]
            },
            {
                "params": [
                    param for name, param in model.named_parameters()
                    if "backbone" in name and param.requires_grad],
                "lr": cfg.SOLVER.OPTIMIZER.BASE_LR_RATIO_BACKBONE * lr,
            },
        ]

        optimizer = maybe_add_full_model_gradient_clipping(optim.AdamW)(
            param_dicts,
            lr=lr,
            betas=cfg.SOLVER.OPTIMIZER.BETAS,
            weight_decay=cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY,
            amsgrad=cfg.SOLVER.OPTIMIZER.AMSGRAD
        )
        return optimizer