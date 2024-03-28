# 导入顺序：Python内置模块、第三方库、本地应用/库
import os
import sys
from multiprocessing import Pool
import warnings


import h5py
import numpy as np
from scipy import io
import glob
from sklearn.preprocessing import MinMaxScaler

# 如果在jupyter中运行，则打开注释，因为运行本文件之前不会运行__init__.py
current_directory = os.getcwd()
SRC = os.path.abspath(os.path.join(current_directory, '../../..'))
if SRC not in sys.path:
    sys.path.append(SRC)

from src.datasets.raw_data import mstar
from src.options import parse_data_args


def generate_data(src_data_path, processed_data_path, dataset_name, mode, use_phase, chip_size):
    """
    将未加工的数据转化为可处理的结构化数据
    """
    # 检查数据是否存在
    if not os.path.exists(src_data_path):
        warnings.warn(f'指定的src_data_path({src_data_path}) 不存在。', UserWarning)
        return

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path, exist_ok=True)

    mstar_dataset = mstar.MSTAR(
        dataset_name=dataset_name, mode=mode, use_phase=use_phase, chip_size=chip_size
    )

    # src_data_path下的所有文件
    image_path_list = glob.glob(os.path.join(src_data_path, '*'))

    azimuths = np.array([]) # 方位角
    meta_label = {}         # 图像的一些属性
    # 根据是否使用相位信息调整总通道数
    # total_images_data_channels = 2 * len(image_path_list) if use_phase else len(image_path_list)
    # images = np.empty((total_images_data_channels, chip_size, chip_size))  # 图像数据
    if use_phase:
        images = np.empty((len(image_path_list), chip_size, chip_size, 2))  # 图像数据
    else:
        images = np.empty((len(image_path_list), chip_size, chip_size, 1))

    for i, img_path in enumerate(image_path_list):
        image, meta_label = mstar_dataset.read(img_path)
        azimuths = np.append(azimuths, meta_label['azimuth_angle'])
        images[i, :, :, :] = image
        # if use_phase:
        #     images[i, :, :, :] = image
        #     # images[i * 2, :, :] = image[:, :, 0]        # 存储第一个通道的数据
        #     # images[i * 2 + 1, :, :] = image[:, :, 1]    # 存储第二个通道的数据
        # else:
        #     images[i, :, :] = np.squeeze(image)

    # # 振幅数据标准化
    # amplitude_data = images[1].reshape(1, 128, 128)     # 振幅数据在第一个通道
    # amplitude_scaler = MinMaxScaler(feature_range=(0, 1))
    # amplitude_normalized = amplitude_scaler.fit_transform(amplitude_data.reshape(-1, 1)).reshape(amplitude_data.shape)
    #
    # # 相位数据标准化
    # phase_data = images[0].reshape(1, 128, 128)         # 相位数据在第二个通道
    # phase_data = np.sqrt(phase_data)                    # 开根号，使数据更集中
    # phase_scaler = MinMaxScaler(feature_range=(0, 1))
    # phase_normalized = phase_scaler.fit_transform(phase_data.reshape(-1, 1)).reshape(phase_data.shape)
    # # max_value = np.amax(phase_data)
    # #
    # # max_index = np.unravel_index(np.argmax(phase_data), phase_data.shape)
    #
    # images = np.squeeze(np.stack((amplitude_normalized, phase_normalized), axis=0))

    # 创建要保存的数据字典(方位角和图像数据)
    images_data = {'azimuths': azimuths, 'images': images}

    # 获取目标类型，即文件夹名字
    target_type = os.path.basename(os.path.normpath(src_data_path))

    # 命名格式：目标类型_sn_序列号_俯仰角_degrees.mat
    # images_data_name = f'{meta_label.target_type}_sn_{meta_label.serial_number}_{meta_label.desired_depression}_degrees.mat'
    images_data_name = f'{target_type}_sn_{meta_label["serial_number"]}_{meta_label["desired_depression"]}_degrees.mat'

    # 保存数据到MAT文件
    with h5py.File(os.path.join(processed_data_path, images_data_name), 'w') as f:
        f.create_dataset('azimuth', data=images_data['azimuths'])
        f.create_dataset('images', data=images_data['images'])


def main():
    # check_requirements(ROOT / 'requirements.txt')

    # 解析输入的命令行参数
    args_input = parse_data_args()
    args = [
        (
            os.path.join(args_input.dataset_path, args_input.mode, target_type),
            os.path.join(args_input.processed_data_path, args_input.mode),
            args_input.dataset_name, args_input.mode, args_input.use_phase, args_input.chip_size
        ) for target_type in mstar.target_name[args_input.dataset_name]
    ]

    for i, arg in enumerate(args):
        print(f'第{i + 1}个。。。')
        generate_data(*arg)

    # 调用多线程
    # with Pool(10) as p:
    #     p.starmap(generate_data, args)


if __name__ == '__main__':
    main()
