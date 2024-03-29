# 导入顺序：Python内置模块、第三方库、本地应用/库
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset

from .dataset import CustomDataset

def load_dataset(data_path, is_train, dataset_name, patch_size, batch_size, train_ratio=0.8):
    """
    加载数据集并返回相应的数据加载器
    """
    # 定义transform
    transform = transforms.Compose([transforms.ToTensor()])

    # 通过自定义数据类获取数据
    _dataset = CustomDataset(data_path=data_path, dataset_name=dataset_name, is_train=is_train, patch_size=patch_size,
                             transform=transform)

    if is_train:
        # # 初始化存储各个子集的列表
        # train_datasets = []
        # valid_datasets = []
        #
        # # 定义每个类别的样本数
        # lengths = [3375, 3375, 3375, 3375, 3375, 3375, 3375, 3375, 3375, 3375]
        # class_item_data_list = split_dataset(_dataset, lengths)
        #
        # for item in class_item_data_list:
        #     train_length = int(3375 * 0.8)
        #     val_length = 3375 - train_length
        #     lengths_item = [train_length, val_length]
        #     train_dataset_item, valid_dataset_item = split_dataset(item, lengths_item)
        #
        #     # 将新的子集添加到列表中
        #     train_datasets.append(train_dataset_item)
        #     valid_datasets.append(valid_dataset_item)
        #
        # # 使用列表中的所有子集创建 ConcatDataset 实例
        # train_dataset = ConcatDataset(train_datasets)
        # valid_dataset = ConcatDataset(valid_datasets)

        # 定义划分方式
        train_length = int(len(_dataset) * 0.8)
        val_length = len(_dataset) - train_length
        lengths = [train_length, val_length]

        # 使用自定义函数划分数据集
        train_dataset, valid_dataset = split_dataset(_dataset, lengths)
        # 划分训练集和验证集
        # train_dataset, valid_dataset = split_train_valid(_dataset, train_ratio)

        _data_loader_train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=is_train, num_workers=0)
        _data_loader_valid = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=is_train, num_workers=0)
        return _data_loader_train, _data_loader_valid
    else:
        _data_loader_test = DataLoader(dataset=_dataset, batch_size=batch_size, shuffle=is_train, num_workers=0)
        return _data_loader_test


def split_train_valid(dataset, train_ratio=0.8):
    """
    根据train_ratio划分训练集和验证集
    """
    total_samples = len(dataset)
    train_size = int(train_ratio * total_samples)
    valid_size = total_samples - train_size
    _train_dataset, _valid_dataset = random_split(dataset, [train_size, valid_size])
    return _train_dataset, _valid_dataset

def split_dataset(dataset, lengths):
    """
    将数据集按照给定的长度列表划分成多个子集
    Args:
        dataset: 要划分的数据集
        lengths: 划分后每个子集的长度列表
    """
    assert sum(lengths) == len(dataset), "Sum of input lengths does not equal the length of the dataset"

    subsets = []
    start_idx = 0
    for length in lengths:
        subset = Subset(dataset, list(range(start_idx, start_idx + length)))
        subsets.append(subset)
        start_idx += length

    return subsets
