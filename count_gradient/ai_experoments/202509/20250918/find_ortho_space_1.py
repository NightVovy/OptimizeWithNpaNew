import sympy as sp
import numpy as np

# 定义符号变量
theta, a0, a1, b0, b1 = sp.symbols('theta a0 a1 b0 b1')

# 定义向量（确保所有元素为sympy表达式）
T1 = [
    sp.sin(2 * theta) * sp.cos(a0),
    sp.sin(2 * theta) * sp.cos(a1),
    sp.sin(2 * theta) * sp.cos(b0),
    sp.sin(2 * theta) * sp.cos(b1),
    -sp.sin(a0) * sp.sin(b0) * sp.cos(2 * theta),
    -sp.sin(a0) * sp.sin(b1) * sp.cos(2 * theta),
    -sp.sin(a1) * sp.sin(b0) * sp.cos(2 * theta),
    -sp.sin(a1) * sp.sin(b1) * sp.cos(2 * theta)
]

T2 = [
    sp.sin(a0) * sp.sin(theta),
    sp.sin(a1) * sp.sin(theta),
    sp.sin(b0) * sp.sin(theta),
    sp.sin(b1) * sp.sin(theta),
    sp.cos(a0) * sp.sin(b0) * sp.cos(theta) - sp.sin(a0) * sp.cos(b0) * sp.sin(theta),
    sp.cos(a0) * sp.sin(b1) * sp.cos(theta) - sp.sin(a0) * sp.cos(b1) * sp.sin(theta),
    sp.cos(a1) * sp.sin(b0) * sp.cos(theta) - sp.sin(a1) * sp.cos(b0) * sp.sin(theta),
    sp.cos(a1) * sp.sin(b1) * sp.cos(theta) - sp.sin(a1) * sp.cos(b1) * sp.sin(theta)
]

T3 = [
    sp.sin(a0) * sp.cos(theta),
    sp.sin(a1) * sp.cos(theta),
    sp.sin(b0) * sp.cos(theta),
    sp.sin(b1) * sp.cos(theta),
    -sp.cos(a0) * sp.sin(b0) * sp.sin(theta) + sp.sin(a0) * sp.cos(b0) * sp.cos(theta),
    -sp.cos(a0) * sp.sin(b1) * sp.sin(theta) + sp.sin(a0) * sp.cos(b1) * sp.cos(theta),
    -sp.cos(a1) * sp.sin(b0) * sp.sin(theta) + sp.sin(a1) * sp.cos(b0) * sp.cos(theta),
    -sp.cos(a1) * sp.sin(b1) * sp.sin(theta) + sp.sin(a1) * sp.cos(b1) * sp.cos(theta)
]

pd_a0 = [
    -sp.cos(2 * theta) * sp.sin(a0),
    sp.sympify(0),
    sp.sympify(0),
    sp.sympify(0),
    -sp.sin(a0) * sp.cos(b0) + sp.sin(2 * theta) * sp.cos(a0) * sp.sin(b0),
    -sp.sin(a0) * sp.cos(b1) + sp.sin(2 * theta) * sp.cos(a0) * sp.sin(b1),
    sp.sympify(0),
    sp.sympify(0)
]

pd_b0 = [
    sp.sympify(0),
    sp.sympify(0),
    -sp.cos(2 * theta) * sp.sin(b0),
    sp.sympify(0),
    -sp.cos(a0) * sp.sin(b0) + sp.sin(2 * theta) * sp.sin(a0) * sp.cos(b0),
    sp.sympify(0),
    -sp.cos(a1) * sp.sin(b0) + sp.sin(2 * theta) * sp.sin(a1) * sp.cos(b0),
    sp.sympify(0)
]

# 代入参数值
values = {
    theta: sp.pi/4,
    a0: 0,
    a1: sp.pi/2,
    b0: sp.pi/4,
    b1: 3 * sp.pi/4
}
# values = {
#     theta: sp.pi/6,
#     a0: 0,
#     a1: 1.42744876,
#     b0: 0.71372438,
#     b1: 2.28452071
# }


# 向量转数值
def vector_to_numeric(vec, values):
    return [float(expr.subs(values).evalf()) for expr in vec]

# 构建矩阵
matrix = np.column_stack([
    vector_to_numeric(T1, values),
    vector_to_numeric(T2, values),
    vector_to_numeric(T3, values),
    vector_to_numeric(pd_a0, values),
    vector_to_numeric(pd_b0, values)
])

# 计算正交补空间
A = matrix.T
U, S, Vh = np.linalg.svd(A, full_matrices=True)
tol = 1e-10
rank = np.sum(S > tol)
null_space_basis = Vh[rank:, :].T
ortho_dim = null_space_basis.shape[1]

# 设置x、y、z的值（可直接修改）
x = 0.33
y = 0.33
z = 0.34

# 计算beta
if ortho_dim >= 3:
    beta = x * null_space_basis[:, 0] + y * null_space_basis[:, 1] + z * null_space_basis[:, 2]
elif ortho_dim == 2:
    beta = x * null_space_basis[:, 0] + y * null_space_basis[:, 1]
elif ortho_dim == 1:
    beta = x * null_space_basis[:, 0]
else:
    beta = np.array([])

# 输出结果
print("空间信息：")
print(f" - 向量空间V的维度: {rank}")
print(f" - 正交补空间的维度: {ortho_dim}\n")

print("正交补空间的基向量（保留8位小数）：")
for i in range(ortho_dim):
    print(f" 基向量{i+1}: {np.round(null_space_basis[:, i], 8)}")

print("\nbeta的一般表达式：")
if ortho_dim >= 3:
    print(" beta = x*基向量1 + y*基向量2 + z*基向量3")
elif ortho_dim == 2:
    print(" beta = x*基向量1 + y*基向量2")
else:
    print(" beta = x*基向量1")

print(f"\n当x={x}, y={y}, z={z}时，beta的值（保留8位小数）：")
if len(beta) > 0:
    beta_rounded = np.round(beta, 8)
    for i in range(len(beta_rounded)):
        print(f" beta[{i+1}] = {beta_rounded[i]}")
