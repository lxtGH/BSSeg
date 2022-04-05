// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <torch/types.h>
#include "ROIAlign/ROIAlign.h"
#include "ROIAlignRotated/ROIAlignRotated.h"
#include "box_iou_rotated/box_iou_rotated.h"
#include "deformable/deform_conv.h"
#include "nms_rotated/nms_rotated.h"
#include "sigmoid_focal_loss/SigmoidFocalLoss.h"
#include "ml_nms/ml_nms.h"
#include "SwapAlign2Nat/SwapAlign2Nat.h"
#include "border_align/border_align.h"
#include "deformable_attn/ms_deform_attn.h"
#include "crop_split/crop_split.h"
#include "crop_split_gt/crop_split_gt.h"
#include "masked_conv2d/masked_conv2d.h"
#include "correlation/correlation.h"

namespace cvpods {

#ifdef WITH_CUDA
extern int get_cudart_version();
#endif

std::string get_cuda_version() {
#ifdef WITH_CUDA
  std::ostringstream oss;

  // copied from
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/CUDAHooks.cpp#L231
  auto printCudaStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
  };
  printCudaStyleVersion(get_cudart_version());
  return oss.str();
#else
  return std::string("not available");
#endif
}

// similar to
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Version.cpp
std::string get_compiler_version() {
  std::ostringstream ss;
#if defined(__GNUC__)
#ifndef __clang__

#if ((__GNUC__ <= 4) && (__GNUC_MINOR__ <= 8))
#error "GCC >= 4.9 is required!"
#endif

  { ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__; }
#endif
#endif

#if defined(__clang_major__)
  {
    ss << "clang " << __clang_major__ << "." << __clang_minor__ << "."
       << __clang_patchlevel__;
  }
#endif

#if defined(_MSC_VER)
  { ss << "MSVC " << _MSC_FULL_VER; }
#endif
  return ss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
  m.def("get_cuda_version", &get_cuda_version, "get_cuda_version");

  m.def("sigmoid_focalloss_forward", &SigmoidFocalLoss_forward, "SigmoidFocalLoss_forward");
  m.def("sigmoid_focalloss_backward", &SigmoidFocalLoss_backward, "SigmoidFocalLoss_backward");

  m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");

  m.def("box_iou_rotated", &box_iou_rotated, "IoU for rotated boxes");

  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
  m.def(
      "deform_conv_backward_input",
      &deform_conv_backward_input,
      "deform_conv_backward_input");
  m.def(
      "deform_conv_backward_filter",
      &deform_conv_backward_filter,
      "deform_conv_backward_filter");
  m.def(
      "modulated_deform_conv_forward",
      &modulated_deform_conv_forward,
      "modulated_deform_conv_forward");
  m.def(
      "modulated_deform_conv_backward",
      &modulated_deform_conv_backward,
      "modulated_deform_conv_backward");

  m.def("deform_unfold_forward", &deform_unfold_forward, "deform unfold forward");
  m.def("deform_unfold_backward_input", &deform_unfold_backward_input, "deform_unfold_backward_input");

  m.def("nms_rotated", &nms_rotated, "NMS for rotated boxes");
  m.def("ml_nms", &ml_nms, "multi-label non-maximum suppression");

  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");

  m.def(
      "roi_align_rotated_forward",
      &ROIAlignRotated_forward,
      "Forward pass for Rotated ROI-Align Operator");
  m.def(
      "roi_align_rotated_backward",
      &ROIAlignRotated_backward,
      "Backward pass for Rotated ROI-Align Operator");
  m.def(
      "swap_align2nat_forward",
      &SwapAlign2Nat_forward,
      "SwapAlign2Nat_forward");
  m.def(
      "swap_align2nat_backward",
      &SwapAlign2Nat_backward,
      "SwapAlign2Nat_backward");
  m.def(
      "border_align_forward",
      &BorderAlign_Forward,
      "BorderAlign_Forward");
  m.def(
      "border_align_backward",
      &BorderAlign_Backward,
      "BorderAlign_Backward");
  m.def(
      "crop_split_forward",
      &crop_split_cuda_forward,
      "crop_split_cuda_forward");
  m.def(
      "crop_split_backward",
      &crop_split_cuda_backward,
      "crop_split_cuda_backward");
  m.def(
      "crop_split_gt_forward",
      &crop_split_gt_cuda_forward,
      "crop_split_gt_cuda_forward");
  m.def(
      "crop_split_gt_backward",
      &crop_split_gt_cuda_backward,
      "crop_split_gt_cuda_backward");

   m.def("masked_im2col_forward",
        &masked_im2col_forward,
        "masked_im2col_forward");
  m.def("masked_col2im_forward",
        &masked_col2im_forward,
        "masked_col2im_forward");

  m.def("corr_forward", &corr_forward, "corr_forward");
  m.def("corr_backward", &corr_backward, "corr_backward");

}

} // namespace cvpods
