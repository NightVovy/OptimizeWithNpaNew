from sympy import symbols, asin, sin, cos, pi, diff, sqrt

# 定义变量
E00, E10, E11, A0, A1, B0, B1 = symbols('E00 E10 E11 A0 A1 B0 B1')

# 定义 theta 和 g
theta = asin((E00 + B0)/(1 + A0)) + asin((E10 + B0)/(1 + A1)) + asin((E11 + B1)/(1 + A1)) - pi
g = sin(theta) * (1 + A0) - B1

# 计算梯度（法向量）
gradient_g = [
    diff(g, E00),
    diff(g, E10),
    diff(g, E11),
    diff(g, A0),
    diff(g, A1),
    diff(g, B0),
    diff(g, B1)
]

print("梯度（法向量）:")
for i, param in enumerate(['E00', 'E10', 'E11', 'A0', 'A1', 'B0', 'B1']):
    print(f"∂g/∂{param} = {gradient_g[i]}")