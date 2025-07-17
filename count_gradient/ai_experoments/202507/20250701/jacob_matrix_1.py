import numpy as np
from sympy import symbols, sin, cos, pi, Matrix

# 定义变量
theta, a0, a1, b0, b1 = symbols('theta a0 a1 b0 b1')


# 手动定义雅可比矩阵
def manual_jacobian(theta_val, a0_val, a1_val, b0_val, b1_val):
    J = np.zeros((8, 5))

    # 填充雅可比矩阵（示例部分，需补全所有40个元素）
    J[0, 0] = -2 * np.sin(2 * theta_val) * np.cos(a0_val)  # ∂A0/∂θ
    J[0, 1] = -np.cos(2 * theta_val) * np.sin(a0_val)  # ∂A0/∂a0
    # ...（补全其他元素）

    return J


# 给定参数值
theta_val = np.pi / 6
a0_val = 0
a1_val = np.pi / 2
b0_val = 1 / np.sqrt(7)
b1_val = 1 / np.sqrt(7)

# 计算手动雅可比矩阵
J_manual = manual_jacobian(theta_val, a0_val, a1_val, b0_val, b1_val)

# 求解法向量
U, S, Vt = np.linalg.svd(J_manual.T)  # J.T是5×8，S长度为5
rank = np.sum(S > 1e-10)  # 矩阵的秩
null_space = Vt[rank:]  # 取最后8-rank行（对应零奇异值）
print("法空间的基向量（行向量）：\n", null_space)