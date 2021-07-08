import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr

__all__ = ['TinyResNet']


class ConvBNLayer(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            groups=1,
            is_vd_mode=False,
            act=None,
            name=None, ):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=stride, stride=stride, padding=0, ceil_mode=True)
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1 if is_vd_mode else stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self._batch_norm = nn.BatchNorm(
            out_channels,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            act='swish',
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act=None,
            name=name + "_branch2b")

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                is_vd_mode=not if_first and stride[0] != 1,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv1)
        y = F.relu(y)
        return y


class TinyResNet(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 depth=None,
                 num_channels=None,
                 num_filters=None,
                 bc_name='tiny_resnet',
                 **kwargs):
        super(TinyResNet, self).__init__()

        depth = [1, 1, 1, 1] if depth is None else depth
        self.num_channels = [32, 32, 64, 128] if num_channels is None else num_channels
        self.num_filters = [32, 64, 128, 256] if num_filters is None else num_filters
        self.conv1_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='swish',
            name=bc_name + "conv1_1")
        self.pool2d_max = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.block_list = []
        for block in range(len(depth)):
            conv_name = "res" + str(block + 2)
            if block == 0:
                stride = (2, 2)
            elif block == 1:
                stride = (1, 1)
            else:
                stride = (2, 1)
            basic_block = self.add_sublayer(
                'bb_%d_%d' % (block, 0),
                BasicBlock(
                    in_channels=self.num_channels[block],
                    out_channels=self.num_filters[block],
                    stride=stride,
                    shortcut=False,
                    if_first=block == 0,
                    name=bc_name + conv_name))
            self.block_list.append(basic_block)
            self.out_channels = self.num_filters[block]
        self.out_channels = self.num_filters[-1] * 2

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.pool2d_max(y)
        for block in self.block_list:
            y = block(y)
        y = y.reshape([-1, 512, 1, 80])
        return y
