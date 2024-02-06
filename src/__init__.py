import sys
from pathlib import Path

# 将src目录载入sys.path中
FILE = Path(__file__).resolve()  # 获取当前文件所在的目录路径
SRC = FILE.parents[0]            # src目录
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))
