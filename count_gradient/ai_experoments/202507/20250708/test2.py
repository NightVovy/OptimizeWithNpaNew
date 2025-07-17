import numpy as np
from sympy import symbols, sin, cos, pi, diff, Matrix, atan

# 定义变量（恢复5个独立变量）
theta, a0, a1, b0, b1 = symbols('theta a0 a1 b0 b1')

# 定义参数化关系
sin2theta = sin(2 * theta)
A0 = cos(2 * theta) * cos(a0)
A1 = cos(2 * theta) * cos(a1)
B0 = cos(2 * theta) * cos(b0)
B1 = cos(2 * theta) * cos(b1)
E00 = cos(a0) * cos(b0) + sin2theta * sin(a0) * sin(b0)
E01 = cos(a0) * cos(b1) + sin2theta * sin(a0) * sin(b1)
E10 = cos(a1) * cos(b0) + sin2theta * sin(a1) * sin(b0)
E11 = cos(a1) * cos(b1) - sin2theta * sin(a1) * sin(b1)

# 构造参数化向量
x = Matrix([A0, A1, B0, B1, E00, E01, E10, E11])

# 定义参数向量（恢复5个变量）
u = Matrix([theta, a0, a1, b0, b1])

# 自动计算雅可比矩阵（8×5）
J_auto = x.jacobian(u)

# 替换具体参数值（CHSH）
theta_val = pi / 4
b0_val = pi / 4
b1_val = 3 * pi / 4

param_values = {
    theta: theta_val,
    a0: 0,
    a1: pi / 2,
    b0: b0_val,
    b1: b1_val
}

# 计算数值雅可比矩阵
J_num = J_auto.subs(param_values).evalf(8)
J_np = np.array(J_num.tolist(), dtype=np.float64)

# 计算8维空间中的点坐标
p = np.array([float(expr.subs(param_values).evalf(8)) for expr in x])

# 计算秩和法空间
U, S, Vt = np.linalg.svd(J_np.T)  # 对5×8矩阵进行SVD
rank = np.sum(S > 1e-10)
null_space = Vt[rank:]  # 法空间基向量

# 输出结果
print("雅可比矩阵（8行×5列）：")
print("行号  ∂/∂θ          ∂/∂a0          ∂/∂a1          ∂/∂b0          ∂/∂b1")
for i, row in enumerate(J_np):
    print(f"{i:2}", end="  ")
    for val in row:
        print(f"{val:14.8f}", end="")
    print()

print(f"\n雅可比矩阵的秩：{rank} (法空间维度：{8 - rank})")

if rank < 8:
    print("\n法空间基向量（行向量）：")
    var_names = ["A0", "A1", "B0", "B1", "E00", "E01", "E10", "E11"]
    for i, vec in enumerate(null_space):
        print(f"n{i + 1}:", " ".join([f"{x:10.6f}" for x in vec]))

    # 输出切平面方程
    print("\n切平面方程（需同时满足）：")
    for i, n in enumerate(null_space):
        terms = []
        for j in range(8):
            if abs(n[j]) > 1e-6:
                terms.append(f"{n[j]:.6f}*({var_names[j]}-{p[j]:.6f})")
        print(f"平面{i + 1}: " + " + ".join(terms) + " = 0")
else:
    print("\n雅可比矩阵满秩，法空间维度为0")

