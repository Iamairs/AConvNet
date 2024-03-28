# 导入顺序：Python内置模块、第三方库、本地应用/库
from torch import nn

from . import _blocks


class Network(nn.Module):
    def __init__(self, classes, channels, **params) -> None:
        super(Network, self).__init__()
        self.channels = channels
        self.classes = classes
        self.dropout_rate = params.get('dropout_rate', 0.2)
        self._w_init = params.get('w_init', lambda x: nn.init.kaiming_normal_(x, nonlinearity='relu'))
        self._b_init = params.get('b_init', lambda x: nn.init.constant_(x, 0.1))

        # self._layer = nn.Sequential(
        #     _blocks.Conv2DBlock(
        #         shape=[5, 5, self.channels, 16], stride=1, padding=0, activation='relu',
        #         max_pool=True, max_pool_size=2, max_pool_stride=2,
        #         w_init=self._w_init, b_init=self._b_init
        #     ),
        #     # _blocks.CBAM(16),
        #     _blocks.Conv2DBlock(
        #         shape=[5, 5, 16, 32], stride=1, padding=0, activation='relu',
        #         max_pool=True, max_pool_size=2, max_pool_stride=2, deformable_groups=2,
        #         w_init=self._w_init, b_init=self._b_init
        #     ),
        #     # _blocks.CBAM(32),
        #     _blocks.Conv2DBlock(
        #         shape=[6, 6, 32, 64], stride=1, padding=0, activation='relu',
        #         max_pool=True, max_pool_size=2, max_pool_stride=2, deformable_groups=2,
        #         w_init=self._w_init, b_init=self._b_init
        #     ),
        #
        #     nn.Dropout(p=0.5),
        #     # _blocks.CBAM(64),
        #     _blocks.Conv2DBlock(
        #         shape=[5, 5, 64, 128], stride=1, padding=0, activation='relu',
        #         max_pool=False, w_init=self._w_init, b_init=self._b_init, deformable_groups=2,
        #     ),
        #     # _blocks.CBAM(128),
        #
        #     _blocks.Conv2DBlock(
        #         shape=[3, 3, 128, self.classes], stride=1, padding=0,
        #         max_pool=False, w_init=self._w_init, b_init=nn.init.zeros_,  deformable_groups=2,
        #         # 零初始化偏置可以被视为一个初始的“无偏”状态，有助于让网络更容易学习适当的权重来进行分类？
        #     ),
        #     nn.Flatten()
        # )

        # AM-CNN模型：
        # self._layer = nn.Sequential(
        #     _blocks.AMConv2dBlock(
        #         shape=[3, 3, self.channels, 16], stride=1, padding=0, activation='relu',
        #         max_pool=False, max_pool_size=2, max_pool_stride=2,
        #         w_init=self._w_init, b_init=self._b_init
        #     ),
        #     _blocks.AMConv2dBlock(
        #         shape=[7, 7, 16, 32], stride=1, padding=0, activation='relu',
        #         max_pool=False, max_pool_size=2, max_pool_stride=2,
        #         w_init=self._w_init, b_init=self._b_init
        #     ),
        #     _blocks.AMConv2dBlock(
        #         shape=[5, 5, 32, 64], stride=1, padding=0, activation='relu',
        #         max_pool=False, max_pool_size=2, max_pool_stride=2,
        #         w_init=self._w_init, b_init=self._b_init
        #     ),
        #
        #     _blocks.AMConv2dBlock(
        #         shape=[5, 5, 64, 128], stride=1, padding=0, activation='relu',
        #         max_pool=True, max_pool_size=2, max_pool_stride=2,
        #         w_init=self._w_init, b_init=self._b_init
        #     ),
        #     _blocks.AMConv2dBlock(
        #         shape=[5, 5, 128, 256], stride=1, padding=0, activation='relu',
        #         max_pool=True, max_pool_size=2, max_pool_stride=2,
        #         w_init=self._w_init, b_init=self._b_init
        #     ),
        #     _blocks.AMConv2dBlock(
        #         shape=[6, 6, 256, 128], stride=1, padding=0, activation='relu',
        #         max_pool=True, max_pool_size=2, max_pool_stride=2,
        #         w_init=self._w_init, b_init=self._b_init
        #     ),
        #
        #     _blocks.AMConv2dBlock(
        #         shape=[5, 5, 128, 64], stride=1, padding=0, activation='relu',
        #         max_pool=False, max_pool_size=2, max_pool_stride=2,
        #         w_init=self._w_init, b_init=self._b_init
        #     ),
        #
        #     _blocks.Conv2DBlock(
        #         shape=[3, 3, 64, self.classes], stride=1, padding=0,
        #         max_pool=False, w_init=self._w_init, b_init=nn.init.zeros_
        #     ),
        #     nn.Softmax(dim=1),
        #     nn.Flatten()
        # )

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
        x = self._layer(x)
        return x

    #     self._layer = nn.Sequential(
    #         _blocks.Conv2DBlock(
    #             shape=[5, 5, self.channels, 16], stride=1, padding=0, activation='relu',
    #             max_pool=True, max_pool_size=2, max_pool_stride=2, batch_norm=True,
    #             w_init=self._w_init, b_init=self._b_init
    #         ),
    #         _blocks.Conv2DBlock(
    #             shape=[5, 5, 16, 32], stride=1, padding=0, activation='relu',
    #             max_pool=True, max_pool_size=2, max_pool_stride=2, batch_norm=True,
    #             w_init=self._w_init, b_init=self._b_init
    #         ),
    #         _blocks.Conv2DBlock(
    #             shape=[6, 6, 32, 64], stride=1, padding=0, activation='relu',
    #             max_pool=True, max_pool_size=2, max_pool_stride=2, batch_norm=True,
    #             w_init=self._w_init, b_init=self._b_init
    #         ),
    #         _blocks.Conv2DBlock(
    #             shape=[5, 5, 64, 128], stride=1, padding=0, activation='relu',
    #             max_pool=False, w_init=self._w_init, b_init=self._b_init
    #         ),
    #     )
    #     self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, dropout=self.dropout_rate)
    #     self.dropout = nn.Dropout(p=self.dropout_rate)
    #     self.fc = nn.Linear(64, self.classes)

    # def forward(self, x):
    #     x = self._layer(x)
    #     # 将卷积层的输出调整为适合LSTM的形状
    #     x = x.view(x.size(0), -1, 128)
    #     x, _ = self.lstm(x)
    #     # 取最后一个时间步
    #     x = x[:, -1, :]
    #     x = self.dropout(x)
    #     x = self.fc(x)

    #     return x
