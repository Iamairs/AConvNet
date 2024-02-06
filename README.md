# 项目名称

## 项目描述

AConvNet是一个基于卷积神经网络的目标分类模型，旨在实现论文《Target Classification Using the Deep Convolutional Networks for SAR Images》中描述的算法。该项目提供了用于训练和测试AConvNet模型的代码，并包含了模型的超参数设置。

## 安装指南

#### 克隆项目仓库：

```bash
git clone https://github.com/Iamairs/AConvNet.git
```

#### 进入项目目录：

```bash
cd AConvNet
```

#### 安装依赖：

```bash
pip install -r requirements.txt
```

#### 数据集准备：

1. 下载数据集[dataset.zip](https://github.com/jangsoopark/AConvNet-pytorch/releases/download/v2.2.0/dataset.zip)文件。
2. 解压文件后，在raw目录下可以找到train和test目录。
3. 将这两个目录（train和test）放置在data/soc/raw目录下。

## 使用说明

### 处理数据

```bash
cd src/datasets/raw_data
python data_entry.py --mode=train
python data_entry.py --mode=test
```

#### 训练模型

```bash
cd ../../
python train.py
```

你可以通过 `--help` 选项查看更多可用参数的说明。

#### 测试模型

```bash
python test.py
```

你可以通过 `--help` 选项查看更多可用参数的说明。

## 项目结构

```
AConvNet/
│
├── data/
│   ├── soc/
│
├── notebook/
│   ├── experiment-soc.py
│
├── outputs/
│   ├── checkpoints/
│   ├── images/
│   ├── logs/
│
├── src/
│   ├── datasets/
│   │   ├── processed_data/
│   │   │   ├── data_loader.py
│   │   │   ├── dataset.py
│   │   ├── raw_data/
│   │   │   ├── data_entry.py
│   │   │   ├── mstar.py
│   ├── model/
│   │   ├── _base.py
│   │   ├── _block.py
│   │   ├── network.py
│   ├── utils/
│   │   ├── general.py
│   │   ├── metrics.py
│   ├── options.py
│   ├── test.py
│   ├── train.py
├── README.md
└── requirements.txt
```


## 参考资料

- [AConvNet 论文链接](https://ieeexplore.ieee.org/document/7460942/)
- [jangsoopark的GitHub仓库](https://github.com/jangsoopark/AConvNet-pytorch)

