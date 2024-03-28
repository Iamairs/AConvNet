# 导入顺序：Python内置模块、第三方库、本地应用/库
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

# 激活函数
_activations = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'leaky_relu': nn.LeakyReLU
}


class BaseBlock(nn.Module):
    def __init__(self):
        super(BaseBlock, self).__init__()
        self._layer: nn.Sequential  # 规定_layer的类型

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

        _bn = params.get('batch_norm')  # 获取批归一化参数
        _act_name = params.get('activation')  # 获取激活函数的名称
        _max_pool = params.get('max_pool')  # 获取最大池化层的参数
        _w_init = params.get('w_init')  # 获取权重初始化函数
        _b_init = params.get('b_init')  # 获取偏置初始化函数

        if _bn:
            _seq.update({'bn': nn.BatchNorm2d(out_channels)})  # 添加批归一化层到层序列中
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
            _b_init(self._layer[idx].bias)  # 初始化卷积层的偏置


class DeformConv2d(nn.Module):
    """
    可变形卷积
    """

    def __init__(self, shape, stride, padding, deformable_groups, **params):
        super(DeformConv2d, self).__init__()
        self.padding = (padding, padding)

        # 获取卷积核的高度、宽度、输入通道数和输出通道数
        h, w, in_channels, out_channels = shape

        # deformable_groups的值必须能被输入通道数整除，其值越大，越复杂
        self.offset_conv = nn.Conv2d(in_channels, 2 * h * w * deformable_groups, kernel_size=h, stride=stride, padding=padding)
        self.modulator_conv = nn.Conv2d(in_channels, h * w * deformable_groups, kernel_size=h, stride=stride, padding=padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=h, stride=stride, padding=padding)

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = torch.sigmoid(self.modulator_conv(x))

        out = deform_conv2d(input=x, offset=offset, weight=self.conv.weight, bias=self.conv.bias, padding=self.padding)
        return out


class DConv2dBlock(BaseBlock):
    """
    可变形卷积模块
    """
    def __init__(self, shape, stride, padding, deformable_groups, **params):
        super(DConv2dBlock, self).__init__()

        self.deform_conv = DeformConv2d(shape, stride, padding, deformable_groups)

        # 获取卷积核的高度、宽度、输入通道数和输出通道数
        h, w, in_channels, out_channels = shape

        # 创建一个有序字典，用于存储层序列
        _seq = collections.OrderedDict([
            ('deform_conv', self.deform_conv)
        ])

        _bn = params.get('batch_norm')  # 获取批归一化参数
        _act_name = params.get('activation')  # 获取激活函数的名称
        _max_pool = params.get('max_pool')  # 获取最大池化层的参数
        _w_init = params.get('w_init')  # 获取权重初始化函数
        _b_init = params.get('b_init')  # 获取偏置初始化函数

        if _bn:
            _seq.update({'bn': nn.BatchNorm2d(out_channels)})  # 添加批归一化层到层序列中
        if _act_name:
            _seq.update({_act_name: _activations[_act_name](inplace=True)})  # 添加激活函数到层序列中
        if _max_pool:
            _kernel_size = params.get('max_pool_size', 2)
            _stride = params.get('max_pool_stride', 2)
            _seq.update({'max_pool': nn.MaxPool2d(kernel_size=_kernel_size, stride=_stride)})  # 添加最大池化层到层序列中

        # 创建包含所有层序列的顺序容器
        self._layer = nn.Sequential(_seq)

        # 获取名为 'conv' 的层在子层列表中的索引
        idx = list(dict(self._layer.named_children()).keys()).index('deform_conv')

        if _w_init:
            _w_init(self._layer[idx].weight)  # 初始化卷积层的权重
        if _b_init:
            _b_init(self._layer[idx].bias)  # 初始化卷积层的偏置


class DSConv2dBlock(BaseBlock):
    """
    深度可分离卷积模块
    """
    def __init__(self, shape, stride, padding, **params):
        super(DSConv2dBlock, self).__init__()

        # 获取卷积核的高度、宽度、输入通道数和输出通道数
        h, w, in_channels, out_channels = shape
        _w_init = params.get('w_init')  # 获取权重初始化函数
        _b_init = params.get('b_init')  # 获取偏置初始化函数

        # 创建一个有序字典，用于存储层序列
        self._seq = collections.OrderedDict([
            ('depthwise', nn.Conv2d(in_channels, in_channels, kernel_size=(h, w), stride=stride, padding=padding, groups=in_channels)),
            ('bn1', nn.BatchNorm2d(in_channels)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pointwise', nn.Conv2d(in_channels, out_channels, kernel_size=1)),
            ('bn2', nn.BatchNorm2d(out_channels)),
            ('relu2', nn.ReLU(inplace=True)),
        ])

        # 将层序列转换为神经网络模块
        self._layer = nn.Sequential(self._seq)

        # 初始化权重和偏置
        if _w_init:
            _w_init(self._layer[0].weight)
            _w_init(self._layer[3].weight)
        if _b_init:
            _b_init(self._layer[0].bias)
            _b_init(self._layer[3].bias)


class SpatialAttention(nn.Module):
    """
    空间注意力模块：空间维度不变，压缩通道维度。该模块关注的是目标的位置信息。
    1） 假设输入的数据x是(b,c,w,h)，并进行两路处理。
    2）其中一路在通道维度上进行求平均值，得到的大小是(b,1,w,h)；另外一路也在通道维度上进行求最大值，得到的大小是(b,1,w,h)。
    3） 然后对上述步骤的两路输出进行连接，输出的大小是(b,2,w,h)
    4）经过一个二维卷积网络，把输出通道变为1，输出大小是(b,1,w,h)
    5）将上一步输出的结果和输入的数据x相乘，最终输出数据大小是(b,c,w,h)。
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # 确保kernel_size为奇数
        assert kernel_size in (3, 7), 'kernel_size一定是3或7'
        padding = 3 if kernel_size == 7 else 1

        # 使用一个卷积层来计算注意力分数
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入x的形状为(batch_size, num_channels, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

        # return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1))) * x


class ChannelAttention(nn.Module):
    """
    通道注意力模型: 通道维度不变，压缩空间维度。该模块关注输入图片中有意义的信息。
    1）假设输入的数据大小是(b,c,w,h)
    2）通过自适应平均池化和自适应最大池化，使得输出的大小变为(b,c,1,1)
    3）将池化后的结果展平，大小变为(b,c)
    4）通过两个全连接层（第一个全连接层后有ReLU激活函数），计算出平均池化和最大池化的注意力分数
    5）将平均池化和最大池化的注意力分数相加，然后通过Sigmoid激活函数，得到最终的通道注意力权重，大小为(b,c,1,1)
    6）将上一步输出的结果和输入的数据相乘，输出数据大小是(b,c,w,h)。
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        # 使用两个全连接层来计算注意力分数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入x的形状为(batch_size, num_channels, H, W)
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        max_out = self.max_pool(x).view(x.size(0), -1)
        avg_out = self.fc2(self.relu1(self.fc1(avg_out)))
        max_out = self.fc2(self.relu1(self.fc1(max_out)))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), -1, 1, 1)

        # return self.sigmoid(out).view(x.size(0), -1, 1, 1) * x


class CBAM(nn.Module):
    """
    卷积注意力模块
    """
    def __init__(self, num_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(num_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先进行通道注意力，然后进行空间注意力
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = F.sigmoid(y)
        return x * y.expand_as(x)


class AMConv2dBlock(BaseBlock):
    def __init__(self, shape, stride, padding, max_pool, **params):
        super(AMConv2dBlock, self).__init__()

        # 获取卷积核的高度、宽度、输入通道数和输出通道数
        h, w, in_channels, out_channels = shape
        _act_name = params.get('activation')  # 获取激活函数的名称
        _w_init = params.get('w_init')  # 获取权重初始化函数
        _b_init = params.get('b_init')  # 获取偏置初始化函数

        # 创建一个有序字典，用于存储层序列
        _seq = collections.OrderedDict([
            # (self, shape, stride, padding, ** params):
            # ('ds_conv', DSConv2dBlock(shape, stride=stride, padding=padding, w_init=_w_init, b_init=_b_init)),
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=(h, w), stride=stride, padding=padding)),
            ('bn', nn.BatchNorm2d(out_channels)),
            (_act_name, _activations[_act_name](inplace=True)),
            ('cbam', CBAM(num_channels=out_channels, reduction_ratio=16, kernel_size=3))
        ])



        if max_pool:
            _kernel_size = params.get('max_pool_size', 2)
            _stride = params.get('max_pool_stride', 2)
            _seq.update({'max_pool': nn.MaxPool2d(kernel_size=_kernel_size, stride=_stride)})  # 添加最大池化层到层序列中

        # 创建包含所有层序列的顺序容器
        self._layer = nn.Sequential(_seq)

        # 获取名为 'conv' 的层在子层列表中的索引
        # idx = list(dict(self._layer.named_children()).keys()).index('conv')
        #
        # if _w_init:
        #     _w_init(self._layer[idx].weight)  # 初始化卷积层的权重
        # if _b_init:
        #     _b_init(self._layer[idx].bias)    # 初始化卷积层的偏置





# —————————————————————————————————————————————— PIHA 注意力模块————————————————————————————————————————————————
class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, *args):
        return args[0]


# Squeeze-and-Excitation模块
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # 自适应平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 两个全连接层，中间有ReLU激活函数，最后有Sigmoid激活函数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 对输入进行全局平均池化，然后通过两个全连接层计算每个通道的权重
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # 对输入进行加权
        return x * y.expand_as(x)


# 选择性平均池化模块
class SelectiveAvgPool2d(nn.Module):
    def __init__(self, thresh=0.1):
        super(SelectiveAvgPool2d, self).__init__()
        self.thresh = thresh

    def forward(self, x):
        # 只对绝对值大于阈值的元素进行平均池化
        x_ = abs(x) > self.thresh
        return ((x * x_).sum(dim=-1).sum(dim=-1) / (x_.sum(dim=-1).sum(dim=-1) + 0.000001)).unsqueeze(-1).unsqueeze(-1)


# 物理信息混合注意力模块
class PIHA(nn.Module):
    def __init__(self, part_num, in_channel, down_rate, reduction=2):
        super(PIHA, self).__init__()
        # 数据驱动流的卷积层和SEBlock
        self.conv_S1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.se_S1 = SEBlock(in_channel, 2)
        # 物理驱动流的卷积层
        self.conv_S2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.phy_group_conv = nn.Conv2d(part_num, in_channel, groups=part_num, kernel_size=down_rate + 1,
                                        stride=down_rate, padding=1)
        # 物理驱动流的选择性平均池化模块和两个卷积层
        self.se_S2 = nn.Sequential(
            SelectiveAvgPool2d(0.05),
            nn.Conv2d(in_channel // part_num, in_channel // (part_num * reduction), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // (part_num * reduction), in_channel // part_num, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, ASC_part):
        b, c, h, w = input.size()
        # 数据驱动流
        X1 = self.conv_S1(input)
        out1 = self.se_S1(X1)
        # 物理驱动流
        # 对ASC_part进行卷积，然后与输入的卷积结果相乘
        ASC_part_ = self.phy_group_conv(ASC_part)
        X2 = self.conv_S2(input)
        fuse_ = ASC_part_ * X2
        fuse = fuse_.view(b, self.part_num, c // self.part_num, h, w)  # bs,s,ci,h,w
        # 对每组结果进行选择性平均池化和两个卷积层处理，计算注意力向量
        se_out = []
        for idx in range(self.part_num):
            se_out.append(self.se_S2(fuse[:, idx, :, :, :]))
        SE_out = torch.stack(se_out, dim=1)
        attention_vectors = self.softmax(SE_out)
        # 对物理驱动流的结果进行加权
        out2 = fuse_ * attention_vectors.view(b, -1, 1, 1)
        # 将数据驱动流和物理驱动流的结果相加
        return out1 + out2
