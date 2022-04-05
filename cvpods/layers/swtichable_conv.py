"""
    Implementation of Switchable Deformable Attention / Convolution Layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from cvpods.layers.deform_conv import deform_conv
from cvpods.layers.wrappers import _NewEmptyTensorOp
from cvpods.layers.deform_unfold_module import DeformUnfold
from cvpods.layers.saconv import ConvAWS2d, constant_init, TORCH_VERSION


class SwitchableConv(ConvAWS2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 use_deform=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        assert use_deform == True

        self.use_deform = use_deform
        self.sampler = DeformUnfold((kernel_size, kernel_size), padding=dilation, dilation=dilation)
        self.N = kernel_size * kernel_size
        self.down_ratio = 4
        self.switch = nn.Conv2d(
            self.in_channels, 1, kernel_size=1, stride=stride, bias=True)
        self.key = nn.Conv2d(
            self.in_channels, self.in_channels // self.down_ratio, kernel_size=1, stride=stride, bias=True)
        self.query = nn.Conv2d(
            self.in_channels, self.in_channels // self.down_ratio, kernel_size=1, stride=stride, bias=True)
        if self.use_deform:
            self.offset_s = nn.Conv2d(
                self.in_channels,
                18,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True)
            self.offset_l = nn.Conv2d(
                self.in_channels,
                18,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True)

        self.init_weights()

    def init_weights(self):
        constant_init(self.switch, 0, bias=1)
        if self.use_deform:
            constant_init(self.offset_s, 0)
            constant_init(self.offset_l, 0)

    def forward(self, x):
        # switch
        B, C, H, W = x.size()
        avg_x = F.pad(x, pad=(2, 2, 2, 2), mode='reflect')
        avg_x = F.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        # for deformable convolution
        weight = self._get_weight(self.weight)
        if self.use_deform:
            offset = self.offset_s(avg_x)
            out_conv = deform_conv(x, offset, weight, self.stride, self.padding, self.dilation, self.groups, 1)
        else:
            out_conv = super().conv2d_forward(x, weight)

        # for deformable attention
        if self.use_deform:
            offset = self.offset_l(avg_x)
            query = self.query(x)
            # sample query
            query = self.sampler(query, offset).view(B, C // self.down_ratio, self.N, H*W)  # (B,C', N, H*W)
            # sample value
            value = self.sampler(x, offset).view(B, C, self.N, H*W)  # (B,C, N, H*W)
            key = self.key(x).view(B, C // self.down_ratio, -1).unsqueeze(2)  # (B,C',1, H*W)
            sim = (key * query).sum(1, keepdim=True)  # (B, 1, N,H*W)
            sim_map = F.softmax(sim, 2)  # (B, 1, N,H*W)
            out_atten = (value * sim_map).sum(2).contiguous()
            out_atten = out_atten.view(B, C, H, W)
        else:
            out_atten = super().conv2d_forward(x, weight)

        out = switch * out_conv + (1 - switch) * out_atten

        return out


class SwitchableConvShareOffset(ConvAWS2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 use_deform=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        assert use_deform == True

        self.use_deform = use_deform
        self.sampler = DeformUnfold((kernel_size, kernel_size), padding=dilation, dilation=dilation)
        self.N = kernel_size * kernel_size
        self.down_ratio = 4
        self.switch = nn.Conv2d(
            self.in_channels, 1, kernel_size=1, stride=stride, bias=True)
        self.key = nn.Conv2d(
            self.in_channels, self.in_channels // self.down_ratio, kernel_size=1, stride=stride, bias=True)
        self.query = nn.Conv2d(
            self.in_channels, self.in_channels // self.down_ratio, kernel_size=1, stride=stride, bias=True)
        if self.use_deform:
            self.offset_s = nn.Conv2d(
                self.in_channels,
                18,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True)

        self.init_weights()

    def init_weights(self):
        constant_init(self.switch, 0, bias=1)
        if self.use_deform:
            constant_init(self.offset_s, 0)

    def forward(self, x):
        # switch
        B, C, H, W = x.size()
        avg_x = F.pad(x, pad=(2, 2, 2, 2), mode='reflect')
        avg_x = F.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        # for deformable convolution
        weight = self._get_weight(self.weight)
        if self.use_deform:
            offset = self.offset_s(avg_x)
            out_conv = deform_conv(x, offset, weight, self.stride, self.padding,
                                self.dilation, self.groups, 1)
        else:
            out_conv = super().conv2d_forward(x, weight)

        ori_p = self.padding
        ori_d = self.dilation
        self.padding = tuple(3 * p for p in self.padding)
        self.dilation = tuple(3 * d for d in self.dilation)

        # for deformable attention
        if self.use_deform:
            query = self.query(x)
            # sample query
            query = self.sampler(query, offset).view(B, C // self.down_ratio, self.N, H*W)  # (B,C', N, H*W)
            # sample value
            value = self.sampler(x, offset).view(B, C, self.N, H*W)  # (B,C, N, H*W)
            key = self.key(x).view(B, C // self.down_ratio, -1).unsqueeze(2)  # (B,C',1, H*W)
            sim = (key * query).sum(1, keepdim=True)  # (B, 1, N,H*W)
            sim_map = F.softmax(sim, 2)  # (B, 1, N,H*W)
            out_atten = (value * sim_map).sum(2).contiguous()
            out_atten = out_atten.view(B, C, H, W)
        else:
            out_atten = super().conv2d_forward(x, weight)

        self.padding = ori_p
        self.dilation = ori_d

        out = switch * out_conv + (1 - switch) * out_atten

        return out


class SwithableCondConv(ConvAWS2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 use_deform=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.use_deform = use_deform
        assert self.use_deform == True

        self.switch = nn.Conv2d(
            self.in_channels, 1, kernel_size=1, stride=stride, bias=True)
        self.weight_diff = nn.Parameter(torch.Tensor(self.weight.size()))
        self.pre_context = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=1, bias=True)
        self.post_context = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=1, bias=True)
        self.N = kernel_size * kernel_size
        self.unfold_1 = DeformUnfold(kernel_size, dilation=dilation, padding=padding)
        self.unfold_2 = DeformUnfold(kernel_size, dilation=dilation*3, padding=padding*3)
        self.down_ratio = 4
        self.key = nn.Conv2d(
            self.in_channels, self.in_channels // self.down_ratio, kernel_size=1, stride=stride, bias=True)
        self.query = nn.Conv2d(
            self.in_channels, self.in_channels // self.down_ratio, kernel_size=1, stride=stride, bias=True)

        if self.use_deform:
            self.offset_s = nn.Conv2d(
                self.in_channels,
                18,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True)
            self.offset_l = nn.Conv2d(
                self.in_channels,
                18,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True)
        self.init_weights()

    def init_weights(self):
        constant_init(self.switch, 0, bias=1)
        self.weight_diff.data.zero_()
        constant_init(self.pre_context, 0)
        constant_init(self.post_context, 0)
        if self.use_deform:
            constant_init(self.offset_s, 0)
            constant_init(self.offset_l, 0)

    def forward(self, x):
        B, C, H, W = x.size()
        # pre-context
        avg_x = F.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x
        # switch
        avg_x = F.pad(x, pad=(2, 2, 2, 2), mode='reflect')
        avg_x = F.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        # sac
        weight = self._get_weight(self.weight)
        c_out, c_in, k, k = weight.shape
        if self.use_deform:
            offset = self.offset_s(avg_x)
            x_unfold = self.unfold_1(x, offset).view(B, C, self.N, H*W)
            query = self.query(x)
            # sample query
            query = self.unfold_1(query, offset).view(B, C // self.down_ratio, self.N, H * W)  # (B,C', N, H*W)
            key = self.key(x).view(B, C // self.down_ratio, -1).unsqueeze(2)  # (B,C',1, H*W)
            sim = (key * query).sum(1, keepdim=True)  # (B, 1, N,H*W)
            sim_map = F.softmax(sim, 2)  # (B, 1, N, H*W)
            x_unfold_atten = x_unfold * sim_map  # (B, C, N, H*W)
            x_unfold_atten = x_unfold_atten.permute(0, 3, 1, 2).flatten(2) # (B,H*W, C*N)
            weight_s = weight.view(c_out, -1)  # (C, C*N)
            out_s = torch.einsum('ijk, ck -> ijc', x_unfold_atten, weight_s).permute(0, 2, 1)
            out_s = out_s.view(B, C, H, W)

        else:
            out_s = super().conv2d_forward(x, weight)

        ori_p = self.padding
        ori_d = self.dilation
        self.padding = tuple(3 * p for p in self.padding)
        self.dilation = tuple(3 * d for d in self.dilation)
        weight = weight + self.weight_diff

        if self.use_deform:
            offset = self.offset_l(avg_x)
            x_unfold = self.unfold_2(x, offset).view(B, C, self.N, H * W)
            query = self.query(x)
            # sample query
            query = self.unfold_2(query, offset).view(B, C // self.down_ratio, self.N, H * W)  # (B,C', N, H*W)
            key = self.key(x).view(B, C // self.down_ratio, -1).unsqueeze(2)  # (B,C',1, H*W)
            sim = (key * query).sum(1, keepdim=True)  # (B, 1, N,H*W)
            sim_map = F.softmax(sim, 2)  # (B, 1, N, H*W)
            x_unfold_atten = x_unfold * sim_map  # (B, C, N, H*W)
            x_unfold_atten = x_unfold_atten.permute(0, 3, 1, 2).flatten(2)  # (B,H*W, C*N)
            weight_l = weight.view(c_out, -1)  # (C, C*N)
            out_l = torch.einsum('ijk, ck -> ijc', x_unfold_atten, weight_l).permute(0, 2, 1)
            out_l = out_l.view(B, C, H, W)

        else:
            out_l = super().conv2d_forward(x, weight)

        out = switch * out_s + (1 - switch) * out_l

        self.padding = ori_p
        self.dilation = ori_d
        # post-context
        avg_x = F.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x
        return out




























###############################################################################
###  Warpper Function ######
##############################################################################



class SwithableCondConv2dLayer(SwithableCondConv):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        if x.numel() == 0 and TORCH_VERSION <= (1, 4):
            assert not isinstance(
                self.norm, torch.nn.GroupNorm
            ), "GroupNorm does not support empty inputs in PyTorch <=1.4!"
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x



class SwitchableConv2dLayer(SwithableCondConv):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        if x.numel() == 0 and TORCH_VERSION <= (1, 4):
            assert not isinstance(
                self.norm, torch.nn.GroupNorm
            ), "GroupNorm does not support empty inputs in PyTorch <=1.4!"
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x



class SwitchableConv2dShareOffsetLayer(SwitchableConvShareOffset):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        if x.numel() == 0 and TORCH_VERSION <= (1, 4):
            assert not isinstance(
                self.norm, torch.nn.GroupNorm
            ), "GroupNorm does not support empty inputs in PyTorch <=1.4!"
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x



# i = torch.Tensor(2,512,128,128).cuda()
# l = SwithableCondConv(512, 512, 3, padding=1).cuda()
# o = l(i)
# print(o.shape)


