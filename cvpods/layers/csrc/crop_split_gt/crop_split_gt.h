#pragma once
#include <torch/types.h>

namespace cvpods {

void CropSplitGtForward(const at::Tensor data,
                        const at::Tensor bbox,
                        at::Tensor out,
                        const int height,
                        const int width,
                        const int num_cell,
                        const int num_bbox);

void CropSplitGtBack(const at::Tensor top_grad,
                     const at::Tensor bbox,
                     at::Tensor bottom_grad,
                     const int height,
                     const int width,
                     const int num_cell,
                     const int num_bbox);


void crop_split_gt_cuda_forward(const at::Tensor input,
                                const at::Tensor bbox,
                                at::Tensor out,
                                const int height,
                                const int width,
                                const int num_cell,
                                const int num_bbox)
{
  TORCH_CHECK(input.is_contiguous(), "input tensor has to be contiguous");

  CropSplitGtForward(input, bbox, out, height, width, num_cell, num_bbox);
}

void crop_split_gt_cuda_backward(const at::Tensor out_grad,
                                 const at::Tensor bbox,
                                 at::Tensor bottom_grad,
                                 const int height,
                                 const int width,
                                 const int num_cell,
                                 const int num_bbox)
{
  TORCH_CHECK(out_grad.is_contiguous(), "out_grad tensor has to be contiguous");

  CropSplitGtBack(out_grad, bbox, bottom_grad, height, width, num_cell, num_bbox);
}


}