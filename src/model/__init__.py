# __init__.py文件的作用
# ①标识包目录： 当一个目录中包含了 __init__.py 文件，Python 解释器就会将这个目录视为一个包，而不仅仅是一个普通的目录。这样，你就可以使用 import 语句导入这个包中的模块。
# ②初始化包： 如果 __init__.py 文件中包含了一些代码，这些代码会在包第一次被导入时执行。这可以用于初始化包中的一些设置或变量。
# ③命名空间的声明： 在 __init__.py 文件中，你可以声明一些变量、函数或类，使它们成为包的一部分，从而可以在包的其他模块中被引用。
# 当使用 import model 时，__init__.py 中的代码会被执行

from ._base import *
