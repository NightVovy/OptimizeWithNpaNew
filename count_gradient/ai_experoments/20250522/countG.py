import numpy as np
from sympy import symbols, sin, asin, diff, simplify

# 定义符号变量
A0, A1, B0, B1, E00, E10, E11 = symbols('A0 A1 B0 B1 E00 E10 E11')

# 定义 theta
theta = (
    asin((E00 + B0) / (1 + A0)) +
    asin((E10 + B0) / (1 + A1)) +
    asin((E11 + B1) / (1 + A1)) - np.pi
)

# 定义 g
g = sin(theta) * (1 + A0) - B1

# 计算 g 对各变量的偏导数
partials = {
    'A0': diff(g, A0),
    'A1': diff(g, A1),
    'B0': diff(g, B0),
    'B1': diff(g, B1),
    'E00': diff(g, E00),
    'E10': diff(g, E10),
    'E11': diff(g, E11),
}

# 打印偏导数
for var, expr in partials.items():
    print(f"∂g/∂{var} = {simplify(expr)}")