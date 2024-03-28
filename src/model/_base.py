# 导入顺序：Python内置模块、第三方库、本地应用/库
import torch

from . import network


class AConvNetModel:
    def __init__(self, classes, channels, lr, momentum=0.9, weight_decay=4e-3, **params):
        # 创建卷积神经网络
        self.net = network.Network(
            classes=classes,
            channels=channels,
            dropout_rate=params.get('dropout_rate')
        )

        # 设置计算设备为GPU或CPU
        _device = params.get('device', 'cuda')
        self.device = torch.device('cuda:0' if ((_device == "cuda") & torch.cuda.is_available()) else 'cpu')
        self.net.to(self.device)

        # 学习率相关参数
        self.lr = lr
        self.lr_scheduler = None

        self.lr_step = params.get('lr_step', '50')   # 学习率衰减的步骤
        self.lr_decay = params.get('lr_decay', 0.1)  # 学习率衰减因子

        # 优化器参数
        self.momentum = momentum
        self.weight_decay = weight_decay

        # 创建随机梯度下降（SGD）优化器
        self.optimizer = torch.optim.SGD(
            params=self.net.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        # 如果设置了学习率衰减，创建学习率调度器
        if self.lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=[int(step) for step in self.lr_step.split(',')],
                gamma=self.lr_decay
            )

        # 交叉熵损失
        self.criterion = torch.nn.CrossEntropyLoss()

    def perform_optimization(self, images, labels):
        """
        执行优化步骤，包括前向传播、计算损失、反向传播和更新参数
        """
        predictions = self.net(images.to(self.device))              # 前向传播
        loss = self.criterion(predictions, labels.to(self.device))  # 计算损失
        self.optimizer.zero_grad()  # 梯度清零
        loss.backward()             # 反向传播
        self.optimizer.step()       # 更新参数

        return loss.item()          # 返回损失值

    @torch.no_grad()
    def inference(self, images):
        """
        获取输入图像的模型输出
        """
        return self.net(images.to(self.device))

    def save(self, epoch, model_path):
        """
        保存模型和优化器状态
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        torch.save(checkpoint, model_path)
