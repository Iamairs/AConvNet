# 导入顺序：Python内置模块、第三方库、本地应用/库
import glob
import os
import random
import re
import warnings

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    自定义数据集类
    """
    def __init__(self, data_path, dataset_name='soc', is_train=True, patch_size=None, transform=None):
        self.dataset_name = dataset_name
        self.is_train = is_train
        self.patch_size = patch_size
        self.transform = transform

        self.images = []
        self.labels = []
        self.serial_number = []

        self._load_data(data_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # 创建一个新的 CustomDataset 实例，包含切片指定的元素
            slice_dataset = CustomDataset(data_path="", dataset_name=self.dataset_name, is_train=self.is_train,
                                          patch_size=self.patch_size, transform=self.transform)
            slice_dataset.images = self.images[idx]
            slice_dataset.labels = self.labels[idx]
            slice_dataset.serial_number = self.serial_number[idx]
            return slice_dataset
        else:
            _image = self.images[idx]
            _label = self.labels[idx]
            _serial_number = self.serial_number[idx]

            if self.transform:
                _image = self.transform(_image)

            return _image, _label, _serial_number

    def _load_data(self, data_path):
        """
        加载数据，并进行数据预处理
        """
        mode = 'train' if self.is_train else 'test'

        # 获取所有mat文件（按首字母顺序，以便获取对应的标签值）
        images_path_list = sorted(glob.glob(os.path.join(data_path, f'{self.dataset_name}/{mode}/*.mat')))
        if len(images_path_list) == 0:
            warnings.warn(f'{mode}数据为空！！！', UserWarning)
            return

        # 定义正则表达式模式，获取目标类型(命名格式：目标类型_sn_序列号_俯仰角_degrees.mat)
        pattern = r'^([^_]+)'

        for image_label, images_path in enumerate(images_path_list):
            target_type = re.match(pattern, os.path.basename(images_path)).group(1)

            # 打开MATLAB v7.3及以上版本的 ".mat" 文件
            images_content = h5py.File(images_path, 'r')

            # 提取 'azimuths' 和 'images' 数据
            # azimuths = images_content['azimuths']
            images_data = images_content['images']
            # 获取图像数量和图像大小
            num_images, image_height, image_width = images_data.shape

            # 遍历 images_data 中的每个图像
            for i in range(num_images):
                image = images_data[i, :, :]

                # 判断是否需要裁剪，进行数据扩充
                if self.is_train:
                    cropped_images = []
                    # 随机裁剪20次，再中心裁剪1次，因此每张图片将被裁剪为21张，以扩充数据集
                    for _ in range(20):
                        cropped_images.append(self._random_crop(image, self.patch_size))
                    cropped_images.append(self._center_crop(image, self.patch_size))

                    for cropped_image in cropped_images:
                        self.images.append(cropped_image)
                        self.labels.append(image_label)
                        self.serial_number.append(target_type)
                else:
                    center_cropped_image = self._center_crop(image, self.patch_size)
                    self.images.append(center_cropped_image)
                    self.labels.append(image_label)
                    self.serial_number.append(target_type)

    @staticmethod
    def _crop_image(image, patch_size):
        """
        将大图裁剪为n张指定尺寸的小图（n = (image_h-patch_size+1)*(image_w-patch_size+1)）
        """
        _cropped_images = []
        # 获取图像长宽
        image_h, image_w = image.shape

        if patch_size > min(image_h, image_w):
            return [image]

        # 裁剪窗口滑动距离
        sliding_length = image_h - patch_size + 1
        sliding_width = image_w - patch_size + 1

        for i in range(sliding_length):
            for j in range(sliding_width):
                _cropped_image = image[i:i + patch_size, j:j + patch_size]
                _cropped_images.append(_cropped_image)

        return _cropped_images

    @staticmethod
    def _center_crop(image, size):
        """
        图像中心裁剪
        """
        image_h, image_w = image.shape

        if size > min(image_h, image_w):
            return image

        # 边缘的宽高（左下角点坐标）
        x = (image_w - size) // 2
        y = (image_h - size) // 2

        image_center = image[y: y + size, x: x + size]
        return image_center

    @staticmethod
    def _random_crop(image, size):
        """
        图像随机裁剪
        """
        image_h, image_w = image.shape

        if size > min(image_h, image_w):
            return image

        x = np.random.randint(0, (image_w-size))
        y = np.random.randint(0, (image_h-size))
        return image[y:y+size, x:x+size]
