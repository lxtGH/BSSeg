
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from cvpods import _C

class _CropSplit(Function):
    @staticmethod
    def forward(ctx, data, rois, c):
        height = data.shape[1]
        width = data.shape[2]
        n = data.shape[3]
        ctx.c = c
        ctx.height = height
        ctx.width = width
        ctx.n = n
        ctx.rois = _pair(rois)
        # print(height*width*n)
        output = data.new_zeros(height, width, n)
        _C.crop_split_forward(data, rois, output, height, width, c, n)
        # print('aa',rois[0])

        # if data.requires_grad:
        #     ctx.save_for_backward(data,rois)
        # print(rois.shape)
        # print(data.requires_grad, rois.requires_grad)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # dtata,_ = ctx.saved_tensors

        c = ctx.c
        height = ctx.height
        width = ctx.width
        n = ctx.n
        rois = ctx.rois
        # print('bb', rois[0])
        grad_input = torch.zeros((c*c, height, width, n), dtype=grad_output.dtype, device=grad_output.device)
        # grad_input = torch.zeros_like(data)
        _C.crop_split_backward(grad_output, rois, grad_input, height, width, c, n)
        # print(grad_output.requires_grad,grad_input.requires_grad)

        return grad_input, None, None

crop_split = _CropSplit.apply

class CropSplit(nn.Module):

    def __init__(self, c=2):
        super(CropSplit, self).__init__()
        self.c = c

    def forward(self, data, rois):
        return crop_split(data, rois, self.c)