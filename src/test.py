import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# 如果在jupyter中运行，则打开注释，因为运行本文件之前不会运行__init__.py
# current_directory = os.getcwd()
# SRC = os.path.abspath(os.path.join(current_directory, '..'))  # 假设 src 目录在当前目录的上一级
# if SRC not in sys.path:
#     sys.path.append(SRC)

from src.datasets.processed_data import data_loader
from src.options import parse_model_args
from src.utils.general import ROOT, set_random_seed, plot_confusion_matrix
from src.utils.metrics import compute_loss, validation
import model


def run(data_path, dataset_name, batch_size, patch_size,
        classes, channels, dropout_rate, activation,
        lr, lr_step, lr_decay, momentum, weight_decay, device, checkpoint_path):
    # 模型的保存路径（记得注释掉！！！！！！！！！！！！！！！！！！）
    checkpoint_path = r"E:\code\objectDetection\AConvNet\outputs\checkpoints\soc\epoch-99\model-soc-99.pth"

    # 加载保存的模型
    AConvNet = model.AConvNetModel(
        classes=classes, channels=channels, dropout_rate=dropout_rate, device=device,
        activation=activation, lr=lr, lr_step=lr_step, lr_decay=lr_decay,
        momentum=momentum, weight_decay=weight_decay
    )

    checkpoint = torch.load(checkpoint_path)
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
