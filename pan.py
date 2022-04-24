import torch
import torch.nn as nn
from .backbone import ConvBNLayer, BasicBlock
import torch.nn.functional as F

__all__ = ['CustomCSPPAN']


class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, k, pool_size, act='swish', data_format='NCHW'):
        super(SPP, self).__init__()
        self.pool = []
        self.data_format = data_format
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2)
            self.pool.append(pool)
        self.conv = ConvBNLayer(in_channels, out_channels, k, padding=k // 2)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        if self.data_format == 'NCHW':
            y = torch.concat(outs, dim=1)
        else:
            y = torch.concat(outs, dim=-1)
        y = self.conv(y)
        return y


class CSPStage(nn.Module):
    def __init__(self, block_fn, ch_in, ch_out, n, act='swish', spp=False):
        super(CSPStage, self).__init__()
        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1)
        self.convs = nn.Sequential()
        next_ch_in = ch_mid
        for i in range(n):
            self.convs.add_sublayer(eval(block_fn)(next_ch_in, ch_mid, act=act, shortcut=False))
            if i == (n - 1) // 2 and spp:
                self.convs.add_sublayer(SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNLayer(ch_mid * 2, ch_out, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = torch.concat([y1, y2], dim=1)
        y = self.conv3(y)
        return y


class CustomCSPPAN(nn.Module):
    __shared__ = ['norm_type', 'data_format', 'width_mult', 'depth_mult', 'trt']

    def __init__(self, in_channels=None, out_channels=None, norm_type='bn', act='leaky',
                 stage_fn='CSPStage', block_fn='BasicBlock', stage_num=1, block_num=3, drop_block=False, block_size=3,
                 keep_prob=0.9, spp=False, data_format='NCHW', width_mult=1.0, depth_mult=1.0, trt=False):

        super(CustomCSPPAN, self).__init__()
        if out_channels is None:
            out_channels = [1024, 512, 256]
        if in_channels is None:
            in_channels = [256, 512, 1024]

        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)

        self.num_blocks = len(in_channels)
        self.data_format = data_format
        self._out_channels = out_channels
        in_channels = in_channels[::-1]
        fpn_stages = []
        fpn_routes = []
        ch_pre = 0
        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2
            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_sublayer(eval(stage_fn)(block_fn, ch_in if j == 0 else ch_out, ch_out, block_num, act=act,spp=(spp and i == 0)))
            fpn_stages.append(stage)
            if i < self.num_blocks - 1:
                fpn_routes.append(ConvBNLayer(in_channels=ch_out, out_channels=ch_out // 2, filter_size=1, stride=1, padding=0, ))
            ch_pre = ch_out

        self.fpn_stages = nn.ModuleList(fpn_stages)
        self.fpn_routes = nn.ModuleList(fpn_routes)
        pan_stages = []
        pan_routes = []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(ConvBNLayer(in_channels=out_channels[i + 1], out_channels=out_channels[i + 1], filter_size=3, stride=2,padding=1, ))
            ch_in = out_channels[i] + out_channels[i + 1]
            ch_out = out_channels[i]
            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_sublayer(eval(stage_fn)(block_fn, ch_in if j == 0 else ch_out, ch_out, block_num, act=act, spp=False))
            pan_stages.append(stage)

        self.pan_stages = nn.ModuleList(pan_stages[::-1])
        self.pan_routes = nn.ModuleList(pan_routes[::-1])

    def forward(self, blocks, for_mot=False):
        blocks = blocks[::-1]
        fpn_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                block = torch.concat([route, block], dim=1)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)
            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.interpolate(route, scale_factor=2)

        pan_feats = [fpn_feats[-1], ]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            block = torch.concat([route, block], dim=1)
            route = self.pan_stages[i](block)
            pan_feats.append(route)

        return pan_feats[::-1]
