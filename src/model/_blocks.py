# 导入顺序：Python内置模块、第三方库、本地应用/库
import collections

import torch.nn as nn

# 激活函数
_activations = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'leaky_relu': nn.LeakyReLU
}


class BaseBlock(nn.Module):
    def __init__(self):
        super(BaseBlock, self).__init__()
        self._layer: nn.Sequential          # 规定_layer的类型

    def forward(self, x):
        return self._layer(x)


class Conv2DBlock(BaseBlock):
    def __init__(self, shape, stride, padding, **params):
        super(Conv2DBlock, self).__init__()

        # 获取卷积核的高度、宽度、输入通道数和输出通道数
        h, w, in_channels, out_channels = shape

        # 创建一个有序字典，用于存储层序列
        _seq = collections.OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=(h, w), stride=stride, padding=padding))
        ])

        _bn = params.get('batch_norm')        # 获取批归一化参数
        _act_name = params.get('activation')  # 获取激活函数的名称
        _max_pool = params.get('max_pool')    # 获取最大池化层的参数
        _w_init = params.get('w_init')        # 获取权重初始化函数
        _b_init = params.get('b_init')        # 获取偏置初始化函数

        if _bn:
            _seq.update({'bn': nn.BatchNorm2d(out_channels)})                # 添加批归一化层到层序列中
        if _act_name:
            _seq.update({_act_name: _activations[_act_name](inplace=True)})  # 添加激活函数到层序列中
        if _max_pool:
            _kernel_size = params.get('max_pool_size', 2)
            _stride = params.get('max_pool_stride', 2)
            _seq.update({'max_pool': nn.MaxPool2d(kernel_size=_kernel_size, stride=_stride)})  # 添加最大池化层到层序列中

        # 创建包含所有层序列的顺序容器
        self._layer = nn.Sequential(_seq)

        # 获取名为 'conv' 的层在子层列表中的索引
        idx = list(dict(self._layer.named_children()).keys()).index('conv')

        if _w_init:
            _w_init(self._layer[idx].weight)  # 初始化卷积层的权重
        if _b_init:
            _b_init(self._layer[idx].bias)    # 初始化卷积层的偏置

