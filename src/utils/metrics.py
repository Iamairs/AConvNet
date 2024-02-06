import torch
from tqdm import tqdm


@torch.no_grad()
def compute_loss(self, images, labels):
    """
    只用于计算损失
    """
    predictions = self.net(images.to(self.device))              # 前向传播
    loss = self.criterion(predictions, labels.to(self.device))  # 计算损失
    return loss.item()                                          # 返回损失值


@torch.no_grad()
def validation(model_temp, data_set):
    """
    测试当前模型效果，并获取分类准确率
    """
    total_num = 0           # 总样本数
    corrects_num = 0        # 正确分类数

    model_temp.net.eval()   # 设置模型为评估模式

    _softmax = torch.nn.Softmax(dim=1)

    for i, data in enumerate(tqdm(data_set)):
        images, labels, _ = data

        predictions = model_temp.inference(images.float())  # 模型推理
        predictions = _softmax(predictions)                 # 使用Softmax函数获取概率分布
        # 获取索引，即预测的类别
        _, predictions = torch.max(predictions.detach(), 1)

        labels = labels.type(torch.LongTensor)
        total_num += labels.size(0)
        corrects_num += (predictions == labels.to(model_temp.device)).sum().item()

    # 计算分类准确率
    _accuracy = 100 * corrects_num / total_num

    return _accuracy
