# 导入顺序：Python内置模块、第三方库、本地应用/库
import os
import json
import logging
import sys

import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold

# 如果在jupyter中运行，则打开注释，因为运行本文件之前不会运行__init__.py
current_directory = os.getcwd()
SRC = os.path.abspath(os.path.join(current_directory, '..'))  # 假设 src 目录在当前目录的上一级
if SRC not in sys.path:
    sys.path.append(SRC)

from src.utils.general import set_random_seed, setup_logger
from src.utils import validation
from src.options import parse_model_args
from src.datasets.processed_data import data_loader
from src.datasets.processed_data.dataset import CustomDataset
import model


def run(data_path, dataset_name, train_ratio, batch_size, patch_size,
        random_seed, classes, channels, dropout_rate, activation,
        lr, lr_step, lr_decay, momentum, weight_decay,
        epochs, output_dir, log_dir, device,
        resume, checkpoint_path):
    """
    训练模型并保存训练结果
    """

    # # 通过自定义数据类获取数据
    # _dataset = CustomDataset(data_path=data_path, dataset_name=dataset_name, is_train=True, patch_size=patch_size,
    #                          transform=transforms.Compose([transforms.ToTensor()]))
    #
    # kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    # step_k = 1
    #
    # for train_index, val_index in kf.split(_dataset):
    #     # print(train_index, test_index)
    #
    #     train_fold = Subset(_dataset, train_index)
    #     valid_fold = Subset(_dataset, val_index)
    #
    #     # 打包成DataLoader类型 用于 训练
    #     train_set = DataLoader(dataset=train_fold, batch_size=batch_size, shuffle=True, num_workers=0)
    #     valid_set = DataLoader(dataset=valid_fold, batch_size=batch_size, shuffle=True, num_workers=0)




    # 加载数据集
    train_set, valid_set = data_loader.load_dataset(data_path=data_path, is_train=True, dataset_name=dataset_name,
                                                    patch_size=patch_size, batch_size=batch_size,
                                                    train_ratio=train_ratio)

    print("[info] 数据加载完成")

    # 初始化AConvNet模型
    AConvNet = model.AConvNetModel(
        classes=classes, channels=channels, dropout_rate=dropout_rate, device=device,
        activation=activation, lr=lr, lr_step=lr_step, lr_decay=lr_decay,
        momentum=momentum, weight_decay=weight_decay
    )
    print("[info] 模型构建完成")

    start_epoch = 1
    # 如果需要从之前的某个时间点开始训练
    if resume:
        checkpoint = torch.load(checkpoint_path)  # 加载checkpoint
        AConvNet.net.load_state_dict(checkpoint['model_state_dict'])  # 恢复模型
        AConvNet.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 恢复优化器状态
        start_epoch = checkpoint['epoch']+1  # 初始训练回合

    # 模型参数和结果保存位置
    model_path = os.path.join(output_dir, dataset_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    for epoch in range(start_epoch, epochs+1):
        train_result = {
            'loss': 0.0,
            'accuracy': 0.0
        }
        _loss = []

        # 训练模型
        AConvNet.net.train()

        # 遍历数据集
        for i, data in enumerate(tqdm(train_set)):
            images, labels, _ = data
            _loss.append(AConvNet.perform_optimization(images.float(), labels))  # 计算损失

        if AConvNet.lr_scheduler:
            lr = AConvNet.lr_scheduler.get_last_lr()[0]
            AConvNet.lr_scheduler.step()

        # 在验证集上评估性能
        accuracy = validation(AConvNet, valid_set)
        train_result['loss'] = np.mean(_loss)
        train_result['accuracy'] = accuracy

        # 保存并打印日志
        # setup_logger(f'{step_k}th_{log_dir}/{dataset_name}_output.log')
        setup_logger(f'{log_dir}/{dataset_name}_output.log')
        logging.info(
            f'Epoch: {epoch:03d}/{epochs:03d} | loss={np.mean(_loss):.4f} | lr={lr} | accuracy={accuracy:.2f}')
        # logging.info(
        #     f'Epoch: {step_k:02d}th | {epoch:03d}/{epochs:03d} | loss={np.mean(_loss):.4f} | lr={lr} | accuracy={accuracy:.2f}')

        # 保存结果
        epoch_path = f'{model_path}/epoch-{epoch}'
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path, True)

        with open(os.path.join(epoch_path, f'history-{dataset_name}-{epoch}.json'), mode='w',
                  encoding='utf-8') as f:
            json.dump(train_result, f, ensure_ascii=True, indent=2)

        AConvNet.save(epoch, os.path.join(epoch_path, f'model-{dataset_name}-{epoch}.pth'))

        # step_k += 1

    print("[info] 模型训练完成！！！")


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
        train_ratio=args.train_ratio,
        random_seed=args.seed,
        classes=args.classes,
        channels=args.channels,
        dropout_rate=args.dropout_rate,
        activation=args.activation,
        lr=args.lr,
        lr_step=args.lr_step,
        lr_decay=args.lr_decay,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        device=args.device,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path,
        )


if __name__ == '__main__':
    main()
