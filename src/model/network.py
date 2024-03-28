# 导入顺序：Python内置模块、第三方库、本地应用/库
from torch import nn

from . import _blocks


class Network(nn.Module):
    def __init__(self, classes, channels, **params) -> None:
        super(Network, self).__init__()
        self.channels = channels
        self.classes = classes
        self.dropout_rate = params.get('dropout_rate', 0.5)
        self._w_init = params.get('w_init', lambda x: nn.init.kaiming_normal_(x, nonlinearity='relu'))
        self._b_init = params.get('b_init', lambda x: nn.init.constant_(x, 0.1))

        self._layer = nn.Sequential(
            _blocks.Conv2DBlock(
                shape=[5, 5, self.channels, 16], stride=1, padding=0, activation='relu',
                max_pool=True, max_pool_size=2, max_pool_stride=2, batch_norm=True,
                w_init=self._w_init, b_init=self._b_init
            ),

            _blocks.Conv2DBlock(
                shape=[3, 3, 16, 32], stride=1, padding=0, activation='relu',
                max_pool=False, max_pool_size=2, max_pool_stride=2, batch_norm=True,
                w_init=self._w_init, b_init=self._b_init
            ),
            _blocks.AMConv2dBlock(
                shape=[3, 3, 32, 64], stride=1, padding=0, activation='relu',
                max_pool=True, max_pool_size=2, max_pool_stride=2, batch_norm=True,
                w_init=self._w_init, b_init=self._b_init
            ),
            _blocks.AMConv2dBlock(
                shape=[1, 1, 64, 32], stride=1, padding=0, activation='relu',
                max_pool=False, max_pool_size=2, max_pool_stride=2, batch_norm=True,
                w_init=self._w_init, b_init=self._b_init
            ),

            _blocks.AMConv2dBlock(
                shape=[6, 6, 32, 64], stride=1, padding=0, activation='relu',
                max_pool=True, max_pool_size=2, max_pool_stride=2, batch_norm=True,
                w_init=self._w_init, b_init=self._b_init
            ),

            # nn.Dropout(p=self.dropout_rate),

            _blocks.AMConv2dBlock(
                shape=[3, 3, 64, 128], stride=1, padding=0, activation='relu',
                max_pool=False, max_pool_size=2, max_pool_stride=2, batch_norm=True,
                w_init=self._w_init, b_init=self._b_init
            ),
            _blocks.AMConv2dBlock(
                shape=[3, 3, 128, 256], stride=1, padding=0, activation='relu',
                max_pool=False, max_pool_size=2, max_pool_stride=2, batch_norm=True,
                w_init=self._w_init, b_init=self._b_init
            ),
            _blocks.Conv2DBlock(
                shape=[1, 1, 256, 128], stride=1, padding=0, activation='relu',
                max_pool=False, max_pool_size=2, max_pool_stride=2, batch_norm=True,
                w_init=self._w_init, b_init=self._b_init
            ),

            _blocks.Conv2DBlock(
                shape=[3, 3, 128, self.classes], stride=1, padding=0,
                max_pool=False, w_init=self._w_init, b_init=nn.init.zeros_
                # 零初始化偏置可以被视为一个初始的“无偏”状态，有助于让网络更容易学习适当的权重来进行分类
            ),
            nn.Flatten()
        )

    def forward(self, x):
        return self._layer(x)
