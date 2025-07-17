import numpy as np
from sympy import symbols, sin, cos, pi, diff, Matrix, sqrt, N

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
J_auto = x.jacobian(u)

# 替换具体参数值
param_values = {
    theta: pi/6,
    a0: 0,
    a1: pi/2,
    b0: 1/sqrt(7),
    b1: 1/sqrt(7)
}

# 计算8维空间中的点坐标
p = [expr.subs(param_values).evalf(8) for expr in x]

# 计算数值雅可比矩阵
J_num = J_auto.subs(param_values).evalf(8)
J_np = np.array(J_num.tolist(), dtype=np.float64)

# 计算秩和法空间
U, S, Vt = np.linalg.svd(J_np.T)  # 注意是J的转置
rank = np.sum(S > 1e-10)
null_space = Vt[rank:]

print("8维空间点坐标：")
var_names = ["A0", "A1", "B0", "B1", "E00", "E01", "E10", "E11"]
for name, val in zip(var_names, p):
    print(f"{name}: {val:.8f}")

print("\n雅可比矩阵（8×5）：")
np.set_printoptions(precision=8, suppress=True)
print(J_np)

print(f"\n雅可比矩阵的秩：{rank}")
print(f"法空间维度：{8-rank}")

print("\n法空间基向量：")
for i, vec in enumerate(null_space):
    print(f"n{i+1}:", " ".join([f"{x:10.8f}" for x in vec]))

# 构造切平面方程
print("\n切平面方程：")
for i, n in enumerate(null_space):
    # 构建方程：n·(X - p) = 0
    equation = " + ".join([f"({n[j]:.8f}*(x{j}-{float(p[j]):.8f}))" for j in range(8)])
    print(f"平面{i+1}: {equation} = 0")

# 更简洁的数学形式表示
print("\n切平面方程（数学形式）：")
for i, n in enumerate(null_space):
    terms = []
    for j in range(8):
        if abs(n[j]) > 1e-6:  # 忽略接近0的系数
            terms.append(f"{n[j]:.8f}({var_names[j]}-{float(p[j]):.8f})")
    eq = " + ".join(terms)
    print(f"平面{i+1}: {eq} = 0")