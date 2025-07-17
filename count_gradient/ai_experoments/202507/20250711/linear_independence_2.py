import sympy as sp
import numpy as np
from sympy import sin, cos, pi, asin

# 定义符号变量
theta, a0, a1, b0, b1 = sp.symbols('theta a0 a1 b0 b1')
s, t = sp.symbols('s t')

# 定义参数
A0 = cos(2 * theta) * cos(a0)
A1 = cos(2 * theta) * cos(a1)
B0 = cos(2 * theta) * cos(b0)
B1 = cos(2 * theta) * cos(b1)
E00 = cos(a0) * cos(b0) + sin(2 * theta) * sin(a0) * sin(b0)
E01 = cos(a0) * cos(b1) + sin(2 * theta) * sin(a0) * sin(b1)
E10 = cos(a1) * cos(b0) + sin(2 * theta) * sin(a1) * sin(b0)
E11 = cos(a1) * cos(b1) + sin(2 * theta) * sin(a1) * sin(b1)

# 定义原始方程
def original_equation(s_val, t_val):
    term1 = asin((E00 + s_val * B0) / (1 + s_val * A0))
    term2 = asin((E01 + s_val * B1) / (1 + s_val * A0))
    term3 = asin((E10 + t_val * B0) / (1 + t_val * A1))
    term4 = -asin((E11 + t_val * B1) / (1 + t_val * A1))
    equation = term1 + term2 + term3 + term4 - pi
    return equation

# 四种 (s, t) 组合
combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

# 生成四个方程
equations = [original_equation(s_val, t_val) for s_val, t_val in combinations]

# 数值化验证
def numerical_check(theta_val, a0_val, a1_val, b0_val, b1_val):
    # 替换具体数值
    subs_dict = {
        theta: theta_val,
        a0: a0_val,
        a1: a1_val,
        b0: b0_val,
        b1: b1_val
    }
    # 计算四个方程的数值
    numerical_equations = [eq.subs(subs_dict).evalf() for eq in equations]
    # 检查是否满足 |equation - pi| < 1e-6
    valid = True
    for i, eq in enumerate(numerical_equations, 1):
        error = abs(eq - pi)
        if error > 1e-6:
            valid = False
            print(f"方程 {i} 不满足条件: 误差 = {error}")
        else:
            print(f"方程 {i} 满足条件: 误差 = {error}")
    if valid:
        print("所有方程均满足 |equation - pi| < 1e-6")
    else:
        print("部分方程不满足条件")
    # 检查线性无关性
    # 构造矩阵（这里假设方程可以线性化）
    # 由于方程是非线性的，我们直接检查数值是否独立
    unique_eqs = list(set([round(float(eq), 6) for eq in numerical_equations]))
    if len(unique_eqs) == 4:
        print("四个方程数值独立")
    else:
        print(f"独立方程数量: {len(unique_eqs)}")

# 选择满足条件的参数值
# 通过解方程或优化方法找到满足条件的参数值
# 这里手动选择一个已知满足条件的参数集？
theta_val = pi/4
a0_val = 0
a1_val = pi/2
b0_val = pi/4
b1_val = 3*pi/4

print("\n参数值：")
print(f"theta = {theta_val}, a0 = {a0_val}, a1 = {a1_val}, b0 = {b0_val}, b1 = {b1_val}")
numerical_check(theta_val, a0_val, a1_val, b0_val, b1_val)