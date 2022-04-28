import torch
import torch.nn as nn


class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3, stride=1, groups=1, padding=0):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=(filter_size, filter_size), stride=(stride, stride), padding=padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class RepVggBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu'):
        super(RepVggBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvBNLayer(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = ConvBNLayer(in_channels, out_channels, 1, stride=1, padding=0)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', shortcut=True):
        super(BasicBlock, self).__init__()
        assert in_channels == out_channels
        self.conv1 = ConvBNLayer(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = RepVggBlock(out_channels, out_channels, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class EffectiveSELayer(nn.Module):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)


class CSPResStage(nn.Module):
    def __init__(self, block_fn, ch_in, ch_out, n, stride, act='relu', attn='eca'):
        super(CSPResStage, self).__init__()
        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(ch_in, ch_mid, 3, stride=2, padding=1)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1)
        self.blocks = nn.Sequential(*[block_fn(ch_mid // 2, ch_mid // 2, act=act, shortcut=True) for _ in range(n)])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
        else:
            self.attn = None
        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.concat([y1, y2], dim=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y


class CSPResNet(nn.Module):
    __shared__ = ['width_mult', 'depth_mult', 'trt']
    def __init__(self, layers=None, channels=None, act='swish', return_idx=None, depth_wise=False, use_large_stem=False,
                 width_mult=1.0, depth_mult=1.0, trt=False):
        super(CSPResNet, self).__init__()
        if return_idx is None:
            return_idx = [0, 1, 2, 3, 4]
        if channels is None:
            channels = [64, 128, 256, 512, 1024]
        if layers is None:
            layers = [3, 6, 6, 3]
        channels = [max(round(c * width_mult), 1) for c in channels]
        layers = [max(round(l * depth_mult), 1) for l in layers]
        act = nn.ReLU(inplace=True)

        if use_large_stem:
            self.stem = nn.Sequential(
                *[ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1),
                  ConvBNLayer(channels[0] // 2, channels[0] // 2, 3, stride=1, padding=1),
                  ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1)])
        else:
            self.stem = nn.Sequential(*[ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1),
                                        ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1)])

        n = len(channels) - 1
        self.stages = nn.Sequential(*[CSPResStage(BasicBlock, channels[i], channels[i + 1], layers[i], 2) for i in range(n)])
        self._out_channels = channels[1:]
        self._out_strides = [4, 8, 16, 32]
        self.return_idx = return_idx

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)

        return outs



if __name__ == '__main__':
    image = torch.randn(1,3,416, 416)
    model_obj = CSPResNet()
    image_output = model_obj(image)
    for x in image_output :
        print(x.shape)
    '''
    
    torch.Size([1, 128, 104, 104])
    torch.Size([1, 256, 52, 52])
    torch.Size([1, 512, 26, 26])
    torch.Size([1, 1024, 13, 13])
    
    '''

