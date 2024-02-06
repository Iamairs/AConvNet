import numpy as np

target_name_soc = (
    '2S1', 'BMP2',
    'BRDM2', 'BTR60',
    'BTR70', 'D7',
    'T62', 'T72',
    'ZIL131', 'ZSU234'
)

# 读取数据集的名称列表
target_name = {
    'soc': target_name_soc,
}


class MSTAR:
    """
    MSTAR类用于读取和处理MSTAR数据集中的图像数据
    """
    def __init__(self, dataset_name='soc', mode='train', use_phase=False, chip_size=128):
        """
        初始化MSTAR类的实例
        """
        self.dataset_name = dataset_name
        self.mode = mode
        self.use_phase = use_phase
        self.chip_size = chip_size

    def read(self, path):
        """
        读取单张图像数据并对其属性进行筛选，生成重构后的数据
        """
        # 返回aut和image
        # 读取单个文件
        with open(path, 'rb') as f:
            _header = self._parse_header(f)         # 读取文件的头部信息
            _data = np.fromfile(f, dtype='>f4')     # 读取二进制数据

        # 获取图片分辨率
        h = eval(_header['NumberOfRows'])
        w = eval(_header['NumberOfColumns'])

        # 按顺序读取具体数据
        _data = _data.reshape(-1, h, w)
        _data = _data.transpose(1, 2, 0)
        _data = _data.astype(np.float32)

        # 修改为单通道，不使用相位数据
        if not self.use_phase:
            _data = np.expand_dims(_data[:, :, 0], axis=2)

        # 进行中心裁剪，将图片更改为128*128大小
        _data = self._center_crop(_data, self.chip_size)

        meta_label = self._extract_meta_label(_header)

        return _data, meta_label

    @staticmethod
    def _parse_header(file):
        """
        解析文件的头部信息
        """
        header = {}
        for line in file:
            line = line.decode('utf-8').strip()

            if not line:
                continue

            if 'PhoenixHeaderVer' in line:
                continue

            if 'EndofPhoenixHeader' in line:
                break

            key_value = line.split('=')
            if len(key_value) == 2:
                key, value = key_value[0].strip(), key_value[1].strip()
                header[key] = value

        return header

    @staticmethod
    def _extract_meta_label(header):
        """
        获取目标类型、序列号、方位角和俯仰角
        """
        target_type = header['TargetType']
        sn = header['TargetSerNum']
        azimuth = eval(header['TargetAz'])
        desired_depression = header['DesiredDepression']

        return {
            'target_type': target_type,
            'serial_number': sn,
            'azimuth_angle': azimuth,
            'desired_depression': desired_depression
        }

    @staticmethod
    def _center_crop(data, size=128):
        """
        图像中心裁剪
        """
        h, w, _ = data.shape

        # 边缘的宽高（左下角点坐标）
        x = (w - size) // 2
        y = (h - size) // 2

        data_center = data[y: y + size, x: x + size]
        return data_center
