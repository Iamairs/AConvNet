import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# 如果在jupyter中运行，则打开注释，因为运行本文件之前不会运行__init__.py
current_directory = os.getcwd()
SRC = os.path.abspath(os.path.join(current_directory, '..'))  # 假设 src 目录在当前目录的上一级
if SRC not in sys.path:
    sys.path.append(SRC)

from src.datasets.processed_data import data_loader
from src.options import parse_model_args
from src.utils.general import ROOT, set_random_seed, plot_confusion_matrix, plot_loss_accuracy
from src.utils.metrics import compute_loss, validation
import model


@torch.no_grad()
def get_labels(model_temp, data_set):
    """
    测试当前模型效果，并获取分类准确率
    """
    _pred_label_list = []
    _true_label_list = []

    model_temp.net.eval()   # 设置模型为评估模式

    _softmax = torch.nn.Softmax(dim=1)

    for i, data in enumerate(tqdm(data_set)):
        images, labels, _ = data

        predictions = model_temp.inference(images.float())  # 模型推理
        predictions = _softmax(predictions)                 # 使用Softmax函数获取概率分布
        # 获取索引，即预测的类别
        _, predictions = torch.max(predictions.detach(), 1)

        labels = labels.type(torch.LongTensor)

        # 将预测标签和真实标签添加到列表中
        _pred_label_list.append(predictions.cpu().numpy())
        _true_label_list.append(labels.cpu().numpy())

    return _pred_label_list, _true_label_list

def generate_noise(_images, _ratio):
    """
    按一定比例向图像中添加噪声
    """
    if _ratio >= 1 or _ratio <= 0:
        print(f"ratio值为：{_ratio}，请设置在0~1范围内")
        return

    n, _, h, w = _images.shape
    noise = np.array([np.random.uniform(size=(1, h, w)) for _ in range(n)])

    # 确定要设置为0的元素的数量，非0元素即为噪声
    total_elements = noise.size
    num_elements_to_set_zero = int(total_elements * (1 - _ratio))

    # 随机选择要设置为0的元素的坐标
    indices_to_set_zero = np.random.choice(total_elements, num_elements_to_set_zero, replace=False)

    # 将选定的元素设置为0
    noise.flat[indices_to_set_zero] = 0

    return _images + noise.astype(np.float32)


def noise_test(_model, _data_set, _ratio):
    """
    在添加一定比例噪声的数据集上测试模型效果，并获取分类准确率
    """
    num_data = 0
    corrects = 0

    _model.net.eval()   # 设置模型为评估模式

    _softmax = torch.nn.Softmax(dim=1)

    for i, data in enumerate(tqdm(_data_set)):
        images, labels, _ = data
        images = generate_noise(images, _ratio)

        predictions = _model.inference(images.float())  # 模型推理
        predictions = _softmax(predictions)                 # 使用Softmax函数获取概率分布
        # 获取索引，即预测的类别
        _, predictions = torch.max(predictions.detach(), 1)

        labels = labels.type(torch.LongTensor)
        num_data += labels.size(0)
        corrects += (predictions == labels.to(_model.device)).sum().item()

    accuracy = 100 * corrects / num_data

    return accuracy



def run(data_path, dataset_name, batch_size, patch_size,
        classes, channels, dropout_rate, activation,
        lr, lr_step, lr_decay, momentum, weight_decay, device, checkpoint_path):
    # 模型的保存路径（记得注释掉！！！！！！！！！！！！！！！！！！）
    checkpoint_path = r"E:\code\objectDetection\AConvNet\outputs\checkpoints\soc\epoch-2\model-soc-2.pth"

    # 加载保存的模型
    AConvNet = model.AConvNetModel(
        classes=classes, channels=channels, dropout_rate=dropout_rate, device=device,
        activation=activation, lr=lr, lr_step=lr_step, lr_decay=lr_decay,
        momentum=momentum, weight_decay=weight_decay
    )
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda:0'))
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    AConvNet.net.load_state_dict(checkpoint['model_state_dict'])

    # 将模型设置为评估模式
    AConvNet.net.eval()

    # 加载测试数据集
    test_set = data_loader.load_dataset(data_path=data_path, is_train=False, dataset_name=dataset_name,
                                        patch_size=patch_size, batch_size=batch_size)

    # 初始化变量以存储测试结果
    test_results = {
        'loss': 0.0,
        'accuracy': 0.0
    }
    _loss = []

    # 在测试集上执行推理
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_set)):
            images, labels, _ = data
            # 计算损失
            _loss.append(compute_loss(AConvNet, images.float(), labels))

        accuracy = validation(AConvNet, test_set)

    # 计算并打印平均测试损失
    test_results['loss'] = np.mean(_loss)
    test_results['accuracy'] = accuracy
    print(f'测试损失：{test_results["loss"]:.4f}，测试准确率：{test_results["accuracy"]}')

    # 将测试结果保存到 JSON 文件中
    test_results_path = f"{ROOT}/outputs/test_results.json"  # 用于保存测试结果的路径
    with open(test_results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=True, indent=2)

    print("[info] 模型测试完成！！！")

    # ----------------------------------------- 绘制绘制损失-准确率图像和混淆矩阵 --------------------------------------------
    # 在测试集上执行推理
    pred_label_list, true_label_list = get_labels(AConvNet, test_set)
    all_pred_labels = np.concatenate(pred_label_list)
    all_true_labels = np.concatenate(true_label_list)
    labels_name = sorted(['2S1', 'BMP2', 'BRDM2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU234'])

    plot_loss_accuracy(
        data_set_result_path=r"E:\code\objectDetection\AConvNet\outputs\checkpoints\soc",
        img_save_path=r'E:\code\objectDetection\AConvNet\outputs\images'
    )

    plot_confusion_matrix(all_true_labels, all_pred_labels, labels_name, title="Confusion Matrix", is_norm=True,
                          img_save_path=r'E:\code\objectDetection\AConvNet\outputs\images')



    # --------------------------------------------------- 噪声测试 ------------------------------------------------------
    noise_result = {}  # 测试结果准确率

    for ratio in [0.01, 0.05, 0.10, 0.15]:
        noise_result[ratio] = noise_test(AConvNet, test_set, ratio)
        print(f'ratio = {ratio:.2f}, accuracy = {noise_result[ratio]:.2f}')

def main():
    print("[info] 开始...")

    # 解析命令行参数
    args = parse_model_args()

    # 设置随机数种子
    set_random_seed(args.seed)

    run(data_path=args.data_path,
        dataset_name=args.dataset_name,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        classes=args.classes,
        channels=args.channels,
        dropout_rate=args.dropout_rate,
        activation=args.activation,
        lr=args.lr,
        lr_step=args.lr_step,
        lr_decay=args.lr_decay,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        device=args.device,
        checkpoint_path=args.checkpoint_path
        )


if __name__ == '__main__':
    main()
