import numpy as np
from sympy import symbols, sin, cos, pi, diff, Matrix, sqrt, N, asin

# 定义变量
theta, a0, a1, b0, b1 = symbols('theta a0 a1 b0 b1')

# 定义参数化关系
A0 = cos(2*theta) * cos(a0)
A1 = cos(2*theta) * cos(a1)
B0 = cos(2*theta) * cos(b0)
B1 = cos(2*theta) * cos(b1)
E00 = cos(a0)*cos(b0) + sin(2*theta)*sin(a0)*sin(b0)
E01 = cos(a0)*cos(b1) + sin(2*theta)*sin(a0)*sin(b1)
E10 = cos(a1)*cos(b0) + sin(2*theta)*sin(a1)*sin(b0)
E11 = cos(a1)*cos(b1) - sin(2*theta)*sin(a1)*sin(b1)

# 构造参数化向量
x = Matrix([A0, A1, B0, B1, E00, E01, E10, E11])

# 定义参数向量
u = Matrix([theta, a0, a1, b0, b1])

# 自动计算雅可比矩阵
J_auto = x.jacobian(u)  # 8×5矩阵

# 替换具体参数值并完全转换为数值
J_num = J_auto.subs({
    theta: pi/6,
    a0: 0,
    a1: pi/2,
    b0: asin(sqrt(3)/2),
    b1: asin(sqrt(3)/2)
}).evalf(8)  # 保留8位小数

# 确保完全转换为浮点数
J_np = np.array(J_num.tolist(), dtype=np.float64)

# 设置打印格式
np.set_printoptions(precision=8, suppress=True, linewidth=120)

print("雅可比矩阵（8行×5列）：")
print("行号  ∂/∂θ          ∂/∂a0          ∂/∂a1          ∂/∂b0          ∂/∂b1")
for i, row in enumerate(J_np):
    print(f"{i:2}", end="  ")
    for val in row:
        print(f"{val:14.8f}", end="")
    print()

# 计算秩
rank = np.linalg.matrix_rank(J_np)
print("\n雅可比矩阵的秩：", rank)

# 求解法向量（可选）
if rank < 8:
    U, S, Vt = np.linalg.svd(J_np.T)  # 对J^T进行SVD
    null_space = Vt[rank:]  # 取最后(8-rank)个向量
    print("\n法空间的基向量（行向量）：")
    for i, vec in enumerate(null_space):
        print(f"n{i+1}:", " ".join([f"{x:10.6f}" for x in vec]))
else:
    print("\n雅可比矩阵满秩，法空间维度为0")