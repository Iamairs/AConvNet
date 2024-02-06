# 导入顺序：Python内置模块、第三方库、本地应用/库
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split

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
        # 划分训练集和验证集
        train_dataset, valid_dataset = split_train_valid(_dataset, train_ratio)

        _data_loader_train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=is_train, num_workers=1)
        _data_loader_valid = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=is_train, num_workers=1)
        return _data_loader_train, _data_loader_valid
    else:
        _data_loader_test = DataLoader(dataset=_dataset, batch_size=batch_size, shuffle=is_train, num_workers=1)
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
