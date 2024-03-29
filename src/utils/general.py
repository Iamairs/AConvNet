# 导入顺序：Python内置模块、第三方库、本地应用/库
import glob
import json
from pathlib import Path
import os
import logging
import random
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 获取项目的根目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]


def set_random_seed(random_seed):
    """
    设置随机种子以实现可复现性
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)  # 让显卡产生的随机数一致
    torch.cuda.manual_seed_all(random_seed)  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证
    np.random.seed(random_seed)  # numpy产生的随机数一致
    random.seed(random_seed)

    # CUDA中的一些运算，它通常使用不确定性算法。
    # 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
    torch.backends.cudnn.deterministic = True

    # 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    # 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
    torch.backends.cudnn.benchmark = False


# 日志配置
def setup_logger(log_file):
    """
    设置日志记录器，将日志输出到文件和控制台
    """
    log_dir = os.path.dirname(log_file)

    os.makedirs(log_dir, exist_ok=True)

    handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=handlers
    )


def get_epoch_number_from_json(json_file_path):
    """
    从文件名中提取数字部分
    """
    match = re.search(r'history-soc-(\d+)\.json', json_file_path)
    _epoch = int(match.group(1))
    return _epoch if match else -1


def plot_loss_accuracy(data_set_result_path, img_save_path=None):
    """
    绘制训练损失和准确率的变化过程
    """
    loss_list = []
    accuracy_list = []
    # 构建文件路径模板
    file_template = os.path.join(data_set_result_path, 'epoch-*', 'history-soc-*.json')
    history_results = sorted(glob.glob(file_template), key=get_epoch_number_from_json)
    for history_result_path in history_results:
        with open(history_result_path) as f:
            history_result = json.load(f)
        loss_list.append(history_result['loss'])
        accuracy_list.append(history_result['accuracy'])

    epochs = np.arange(len(loss_list))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    plot1, = ax1.plot(epochs, loss_list, marker='.', c='blue', label='loss')
    plot2, = ax2.plot(epochs, accuracy_list, marker='.', c='red', label='accuracy')
    plt.legend([plot1, plot2], ['loss', 'accuracy'], loc='upper right')

    plt.grid()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('loss', color='blue')
    ax2.set_ylabel('accuracy', color='red')

    if img_save_path:
        img_name = 'loss_accuracy.png'
        img_path = os.path.join(img_save_path, img_name)
        plt.savefig(img_path, bbox_inches='tight', dpi=300)

    plt.show()


def plot_confusion_matrix(label_true, label_pred, labels_name, title="Confusion Matrix", is_norm=True, color_bar=True,
                          img_save_path=None):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred)
    if is_norm:
        # 对混淆矩阵进行归一化，并格式化数值
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)

    plt.imshow(cm, interpolation='nearest', cmap='Blues')   # 在特定的窗口上显示图像
    for i in range(len(cm)):
        for j in range(len(cm)):
            color = (1, 1, 1) if i == j else (0, 0, 0)      # 将对角线值设为黑色
            # cm[j, i]表示的是实际类别是j，预测类别是i的那个位置的值
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', color=color, verticalalignment='center')

    if color_bar:
        plt.colorbar()  # 创建颜色条

    tick_loc = np.array(range(len(labels_name)))    # 刻度位置
    plt.xticks(tick_loc, labels_name, rotation=45)  # 将标签印在x轴坐标上
    plt.yticks(tick_loc, labels_name)               # 将标签印在y轴坐标上
    plt.title(title)                                # 图像标题
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if img_save_path:
        img_name = 'confusion_matrix_norm.png' if is_norm else 'confusion_matrix.png'
        img_path = os.path.join(img_save_path, img_name)
        plt.savefig(img_path, bbox_inches='tight', dpi=300)

    plt.show()                                      # plt.show()需在plt.savefig()之后

def showSarImage(sar_image):
    # 提取幅度和相位信息
    phase = sar_image[0]
    amplitude = sar_image[1]

    # 创建子图
    fig, axs = plt.subplots(2, 1, figsize=(6, 10))

    # 显示幅度信息
    axs[0].imshow(amplitude, cmap='gray')
    axs[0].set_title('Phase')

    # 显示相位信息
    axs[1].imshow(phase, cmap='gray')
    axs[1].set_title('Amplitude')

    # 显示图像
    plt.tight_layout()
    plt.show()
