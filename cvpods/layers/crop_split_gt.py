
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from cvpods import _C

class _CropSplitGT(Function):

    @staticmethod
    def forward(ctx, data, rois, c):
        height = data.shape[0]
        width = data.shape[1]
        n = data.shape[2]
        ctx.c = _pair(c)
        ctx.height = _pair(height)
        ctx.width = _pair(width)
        ctx.n = _pair(n)
        # ctx.rois = rois
        # print(height*width*n)
        output = data.new_zeros(height, width, n)
        _C.crop_split_gt_forward(data, rois, output, height, width, c, n)
        # print('aa',rois[0])

        # ctx.save_for_backward(data,rois)
        # print(torch.max(output_gt))
        # print('aa',output_gt.shape)
        # print(rois.shape)
        # print(data.requires_grad, rois.requires_grad)
        return output

crop_split_gt = _CropSplitGT.apply

class CropSplitGT(nn.Module):

    def __init__(self, c=2):
        super(CropSplitGT, self).__init__()
        self.c = c

    def forward(self, data, rois):
        return crop_split_gt(data, rois, self.c)
