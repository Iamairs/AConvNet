# 导入顺序：Python内置模块、第三方库、本地应用/库
import argparse

from src.utils.general import ROOT


def parse_data_args():
    """
    解析数据处理参数
    """
    parser = argparse.ArgumentParser(description='数据处理参数设置')

    parser.add_argument('--dataset_path', type=str, default=ROOT / 'data/soc/raw', help='原生数据路径')
    parser.add_argument('--processed_data_path', type=str, default=ROOT / 'data/soc', help='处理后的数据路径')
    parser.add_argument('--dataset_name', type=str, default='soc', help='数据集名称')
    parser.add_argument('--use_phase', action='store_true', default=False, help='是否使用相位信息')
    parser.add_argument('--mode', type=str, default='test', help='处理训练集还是测试集')
    parser.add_argument('--chip_size', type=int, default=128, help='图像裁剪尺寸')
    args = parser.parse_args()
    return args


def parse_model_args():
    """
    解析分类模型参数
    """
    parser = argparse.ArgumentParser(description='分类模型参数设置')

    # 数据集参数
    parser.add_argument('--data_path', type=str, default=ROOT / 'data', help='所以的数据所在路径')
    parser.add_argument('--dataset_name', type=str, default='soc', help='数据集名称')
    parser.add_argument('--classes', type=int, default=10, help='类别数量')
    parser.add_argument('--channels', type=int, default=1, help='输入图像通道数量')

    # 训练参数
    parser.add_argument('--patch_size', type=int, default=88, help='训练时的图像块尺寸')
    parser.add_argument('--batch_size', type=int, default=100, help='每个训练批次的图像数量')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集和测试集划分比例')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--activation', type=str, default='relu', help='激活函数')
    parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--lr_step', type=str, default='50', help='学习率调度器的步长参数')    # 注意是字符串！！！
    parser.add_argument('--lr_decay', type=float, default=0.1, help='学习率调度器的衰减因子参数')
    parser.add_argument('--momentum', type=float, default=0.9, help='优化器的动量参数')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减的参数')
    parser.add_argument('--epochs', type=int, default=50, help='训练的轮次数')

    # 其他参数
    parser.add_argument('--output_dir', type=str, default=ROOT / 'outputs/checkpoints', help='推理结果的输出目录')
    parser.add_argument('--log_dir', type=str, default=ROOT / 'outputs/logs', help='保存训练日志的目录')
    parser.add_argument('--checkpoint_path', type=str, default=ROOT / 'outputs/checkpoints/soc/epoch-100/model-soc-100.pth', help='模型所在路径')
    parser.add_argument('--resume', action='store_true', help='是否从先前的训练中恢复模型')
    parser.add_argument('--pretrained', action='store_true', help='是否使用预训练的模型权重')
    parser.add_argument('--evaluate', action='store_true', help='是否只进行模型评估而不进行训练')
    parser.add_argument('--device', type=str, default='cuda', help='运行模型的设备（cuda或cpu）')
    parser.add_argument('--seed', type=int, default=666, help='随机种子')

    args = parser.parse_args()
    return args
