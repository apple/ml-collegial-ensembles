"""
Creates a ResNeXt Model as defined in:

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016).
Aggregated residual transformations for deep neural networks.
arXiv preprint arXiv:1611.05431.

Original author of `ResNeXtBottleneck` and  `CifarResNeXt`: "Pau Rodríguez López, ISELAB, CVC-UAB"
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """
    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """
        :param in_channels: input channel dimensionality
        :param out_channels: output channel dimensionality
        :param stride: conv stride. Replaces pooling layer.
        :param cardinality: num of convolution groups.
        :param base_width: base number of channels in each group.
        :param widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1,
                                   groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0,
                                     bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv', nn.Conv2d(in_channels, out_channels,
                                            kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):
    """
    Building ResNeXt-29 for CIFAR-10/100 following https://arxiv.org/pdf/1611.05431.pdf.
    """
    def __init__(self, cardinality, depth, nlabels, base_width, widen_factor=4):
        """
        :param cardinality: number of convolution groups.
        :param depth: number of layers.
        :param nlabels: number of classes
        :param base_width: base number of channels in each group.
        :param widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(self.stages[3], nlabels)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_in')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """
        Stack n bottleneck modules where n is inferred from the depth of the network.

        :param name: string name of the current block.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        :return: a module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width, self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                   self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.classifier(x)


class SmallImageNetResNeXt(nn.Module):
    """
    Building ResNeXt-29 or 38 for ImageNet32x32 and ImageNet64x64 respectively.
    """
    def __init__(self, cardinality, depth, nlabels, base_width, resolution, widen_factor=4):
        """
        :param cardinality: number of convolution groups.
        :param depth: number of layers.
        :param nlabels: number of classes
        :param base_width: base number of channels in each group.
        :param widen_factor: factor to adjust the channel dimensionality
        """
        super().__init__()
        assert resolution in [16, 32, 64], '{6, 32, 64} are the only supported resolutions.'

        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = int(np.round((self.depth - 2) // 9))
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.resolution = resolution
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor] #, 64 * self.widen_factor]
        if self.resolution < 16:
            self.stages = self.stages[:-1]
        if self.resolution >= 32:
            self.stages += [256 * self.widen_factor]
        if self.resolution >= 64:
            self.stages += [512 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)

        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        if self.resolution >= 32:
            self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        if self.resolution >= 64:
            self.stage_4 = self.block('stage_4', self.stages[3], self.stages[4], 2)

        self.classifier = nn.Linear(self.stages[-1], nlabels)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_in')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """
        Stack n bottleneck modules where n is inferred from the depth of the network.

        :param name: string name of the current block.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        :return: a module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width, self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                   self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)


        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        if self.resolution >= 32:
            x = self.stage_3.forward(x)
        if self.resolution >= 64:
            x = self.stage_4.forward(x)

        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[-1])
        return self.classifier(x)


class ImageNetResNeXt(nn.Module):
    """
    Building ResNeXt-50, 101 or 152 for ImageNet.
    """
    def __init__(self, cardinality, depth, nlabels, base_width):
        """
        :param cardinality: number of convolution groups.
        :param depth: number of layers.
        :param nlabels: number of classes
        :param base_width: base number of channels in each group.
        """
        super().__init__()

        assert depth in [50, 101, 152], "depth should be in {50, 101, 152} but is {}".format(depth)

        self.cardinality = cardinality
        self.depth = depth

        self.cfg = {50: [3, 4, 6, 3],
                    101: [3, 4, 23, 3],
                    152: [3, 8, 36, 3]}

        self.base_width = base_width
        self.widen_factor = 4
        self.nlabels = nlabels
        self.stages = [64,
                       64 * self.widen_factor, # 256
                       128 * self.widen_factor, # 512
                       256 * self.widen_factor, # 1024
                       512 * self.widen_factor] # 2048

        self.conv_1_7x7 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.maxpool_3x3_1 = nn.MaxPool2d(3, 2, 1)

        cfg = self.cfg[self.depth]
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], cfg[0], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], cfg[1], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], cfg[2], 2)
        self.stage_4 = self.block('stage_4', self.stages[3], self.stages[4], cfg[3], 2)

        self.classifier = nn.Linear(self.stages[-1], self.nlabels)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_in')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, count, pool_stride):
        """
        Stack bottleneck modules.

        :param name: string name of the current block.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param count: number of residual blocks to stack.
        :param pool_stride: factor to reduce the spatial dimensionality in the first bottleneck
                    of the block.
        :return: a module consisting of `count` sequential bottlenecks.

        """
        block = nn.Sequential()
        for bottleneck in range(count):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride,
                                            self.cardinality, self.base_width, self.widen_factor))
            else:
                block.add_module(name_, ResNeXtBottleneck(out_channels, out_channels, 1,
                                            self.cardinality, self.base_width, self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_7x7.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.maxpool_3x3_1.forward(x)


        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = self.stage_4.forward(x)

        x = F.avg_pool2d(x, 7, 1)
        x = x.view(-1, self.stages[-1])
        return self.classifier(x)
