import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from torch.nn.modules.utils import _pair
from cvpods.layers.deform_conv import DeformConv, deform_conv


class SEPC(nn.Module):
    def __init__(
            self,
            in_features,
            in_channels,
            out_channels,
            num_outs,
            pconv_deform,
            lcconv_deform,
            iBN,
            Pconv_num,
    ):
        super(SEPC, self).__init__()
        assert isinstance(in_channels, list)
        self.in_features = in_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        assert num_outs == 5
        self.fp16_enabled = False
        self.iBN = iBN
        self.Pconvs = nn.ModuleList()

        for i in range(Pconv_num):
            self.Pconvs.append(
                PConvModule(in_channels[i],
                            out_channels,
                            iBN=self.iBN,
                            part_deform=pconv_deform))

        self.lconv = sepc_conv(256,
                               256,
                               kernel_size=3,
                               dilation=1,
                               padding=1,
                               part_deform=lcconv_deform)
        self.cconv = sepc_conv(256,
                               256,
                               kernel_size=3,
                               dilation=1,
                               padding=1,
                               part_deform=lcconv_deform)
        self.relu = nn.ReLU()
        if self.iBN:
            self.lbn = nn.BatchNorm2d(256)
            self.cbn = nn.BatchNorm2d(256)
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for str in ['l', 'c']:
            m = getattr(self, str + 'conv')
            init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        x = [inputs[f] for f in self.in_features]
        for pconv in self.Pconvs:
            x = pconv(x)
        cls = [self.cconv(level, item) for level, item in enumerate(x)]
        loc = [self.lconv(level, item) for level, item in enumerate(x)]
        if self.iBN:
            cls = iBN(cls, self.cbn)
            loc = iBN(loc, self.lbn)
        outs = [[self.relu(s), self.relu(l)] for s, l in zip(cls, loc)]
        return dict(zip(self.in_features, outs))


class PConvModule(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        kernel_size=[3, 3, 3],
        dilation=[1, 1, 1],
        groups=[1, 1, 1],
        iBN=False,
        part_deform=False,
    ):
        super(PConvModule, self).__init__()

        #     assert not (bias and iBN)
        self.iBN = iBN
        self.Pconv = nn.ModuleList()
        self.Pconv.append(
            sepc_conv(in_channels,
                      out_channels,
                      kernel_size=kernel_size[0],
                      dilation=dilation[0],
                      groups=groups[0],
                      padding=(kernel_size[0] + (dilation[0] - 1) * 2) // 2,
                      part_deform=part_deform))
        self.Pconv.append(
            sepc_conv(in_channels,
                      out_channels,
                      kernel_size=kernel_size[1],
                      dilation=dilation[1],
                      groups=groups[1],
                      padding=(kernel_size[1] + (dilation[1] - 1) * 2) // 2,
                      part_deform=part_deform))
        self.Pconv.append(
            sepc_conv(in_channels,
                      out_channels,
                      kernel_size=kernel_size[2],
                      dilation=dilation[2],
                      groups=groups[2],
                      padding=(kernel_size[2] + (dilation[2] - 1) * 2) // 2,
                      stride=2,
                      part_deform=part_deform))

        if self.iBN:
            self.bn = nn.BatchNorm2d(256)

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.Pconv:
            init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        next_x = []
        for level, feature in enumerate(x):                         # Loop for each fpn layer

            temp_fea = self.Pconv[1](level, feature)
            if level > 0:
                temp_fea += self.Pconv[2](level, x[level - 1])
            if level < len(x) - 1:
                temp_fea += F.upsample_bilinear(
                    self.Pconv[0](level, x[level + 1]),
                    size=[temp_fea.size(2), temp_fea.size(3)])
            next_x.append(temp_fea)
        if self.iBN:
            next_x = iBN(next_x, self.bn)
        next_x = [self.relu(item) for item in next_x]
        return next_x


def iBN(fms, bn):
    sizes = [p.shape[2:] for p in fms]
    n, c = fms[0].shape[0], fms[0].shape[1]
    fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
    fm = bn(fm)
    fm = torch.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
    return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]


class sepc_conv(DeformConv):
    def __init__(self, *args, part_deform=False, **kwargs):
        super(sepc_conv, self).__init__(*args, **kwargs)
        self.part_deform = part_deform
        if self.part_deform:
            self.conv_offset = nn.Conv2d(self.in_channels,
                                         self.deformable_groups * 2 *
                                         self.kernel_size[0] *
                                         self.kernel_size[1],
                                         kernel_size=self.kernel_size,
                                         stride=_pair(self.stride),
                                         padding=_pair(self.padding),
                                         bias=True)
            self.init_offset()

        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        self.start_level = 1

    def init_offset(self):

        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, i, x):
        if i < self.start_level or not self.part_deform:
            return torch.nn.functional.conv2d(x,
                                              self.weight,
                                              bias=self.bias,
                                              stride=self.stride,
                                              padding=self.padding,
                                              dilation=self.dilation,
                                              groups=self.groups)

        offset = self.conv_offset(x)
        return deform_conv(x, offset, self.weight, self.stride, self.padding,
                           self.dilation, self.groups,
                           self.deformable_groups) + self.bias.unsqueeze(
                               0).unsqueeze(-1).unsqueeze(-1)


def build_sepc_head(cfg):
    in_features = cfg.MODEL.SEPC.IN_FEATURES
    in_channels = cfg.MODEL.SEPC.IN_CHANNELS
    out_channels = cfg.MODEL.SEPC.OUT_CHANNELS
    num_outs = cfg.MODEL.SEPC.NUM_OUTS
    combine_head_deform = cfg.MODEL.SEPC.COMBINE_DEFORM
    extra_head_deform = cfg.MODEL.SEPC.EXTRA_DEFORM
    combine_head_num = cfg.MODEL.SEPC.COMBINE_NUM
    iBN = cfg.MODEL.SEPC.IBN
    sepc_head = SEPC(in_features=in_features,
                     in_channels=in_channels,
                     out_channels=out_channels,
                     num_outs=num_outs,
                     pconv_deform=combine_head_deform,
                     lcconv_deform=extra_head_deform,
                     iBN=iBN,
                     Pconv_num=combine_head_num)
    return sepc_head