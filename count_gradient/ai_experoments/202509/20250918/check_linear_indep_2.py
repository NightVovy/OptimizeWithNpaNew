import sympy as sp
import numpy as np

# 定义符号变量
theta, a0, a1, b0, b1 = sp.symbols('theta a0 a1 b0 b1')

# 定义偏导数向量（根据你提供的表达式）
dP_dtheta = sp.Matrix([
    -2 * sp.sin(2 * theta) * sp.cos(a0),
    -2 * sp.sin(2 * theta) * sp.cos(a1),
    -2 * sp.sin(2 * theta) * sp.cos(b0),
    -2 * sp.sin(2 * theta) * sp.cos(b1),
    2 * sp.sin(a0) * sp.sin(b0) * sp.cos(2 * theta),
    2 * sp.sin(a0) * sp.sin(b1) * sp.cos(2 * theta),
    2 * sp.sin(a1) * sp.sin(b0) * sp.cos(2 * theta),
    2 * sp.sin(a1) * sp.sin(b1) * sp.cos(2 * theta)
])

dP_da0 = sp.Matrix([
    -sp.sin(a0) * sp.cos(2 * theta),
    0,
    0,
    0,
    -sp.sin(a0) * sp.cos(b0) + sp.sin(b0) * sp.sin(2 * theta) * sp.cos(a0),
    -sp.sin(a0) * sp.cos(b1) + sp.sin(b1) * sp.sin(2 * theta) * sp.cos(a0),
    0,
    0
])

dP_da1 = sp.Matrix([
    0,
    -sp.sin(a1) * sp.cos(2 * theta),
    0,
    0,
    0,
    0,
    -sp.sin(a1) * sp.cos(b0) + sp.sin(b0) * sp.sin(2 * theta) * sp.cos(a1),
    -sp.sin(a1) * sp.cos(b1) + sp.sin(b1) * sp.sin(2 * theta) * sp.cos(a1)
])

dP_db0 = sp.Matrix([
    0,
    0,
    -sp.sin(b0) * sp.cos(2 * theta),
    0,
    sp.sin(a0) * sp.sin(2 * theta) * sp.cos(b0) - sp.sin(b0) * sp.cos(a0),
    0,
    sp.sin(a1) * sp.sin(2 * theta) * sp.cos(b0) - sp.sin(b0) * sp.cos(a1),
    0
])

dP_db1 = sp.Matrix([
    0,
    0,
    0,
    -sp.sin(b1) * sp.cos(2 * theta),
    0,
    sp.sin(a0) * sp.sin(2 * theta) * sp.cos(b1) - sp.sin(b1) * sp.cos(a0),
    0,
    sp.sin(a1) * sp.sin(2 * theta) * sp.cos(b1) - sp.sin(b1) * sp.cos(a1)
])

# 定义T2和T3向量
T2 = sp.Matrix([
    sp.sin(a0) * sp.sin(theta),
    sp.sin(a1) * sp.sin(theta),
    sp.sin(b0) * sp.sin(theta),
    sp.sin(b1) * sp.sin(theta),
    sp.cos(a0) * sp.sin(b0) * sp.cos(theta) - sp.sin(a0) * sp.cos(b0) * sp.sin(theta),
    sp.cos(a0) * sp.sin(b1) * sp.cos(theta) - sp.sin(a0) * sp.cos(b1) * sp.sin(theta),
    sp.cos(a1) * sp.sin(b0) * sp.cos(theta) - sp.sin(a1) * sp.cos(b0) * sp.sin(theta),
    sp.cos(a1) * sp.sin(b1) * sp.cos(theta) - sp.sin(a1) * sp.cos(b1) * sp.sin(theta)
])

T3 = sp.Matrix([
    sp.sin(a0) * sp.cos(theta),
    sp.sin(a1) * sp.cos(theta),
    sp.sin(b0) * sp.cos(theta),
    sp.sin(b1) * sp.cos(theta),
    -sp.cos(a0) * sp.sin(b0) * sp.sin(theta) + sp.sin(a0) * sp.cos(b0) * sp.cos(theta),
    -sp.cos(a0) * sp.sin(b1) * sp.sin(theta) + sp.sin(a0) * sp.cos(b1) * sp.cos(theta),
    -sp.cos(a1) * sp.sin(b0) * sp.sin(theta) + sp.sin(a1) * sp.cos(b0) * sp.cos(theta),
    -sp.cos(a1) * sp.sin(b1) * sp.sin(theta) + sp.sin(a1) * sp.cos(b1) * sp.cos(theta)
])

# 所有要验证的向量
vectors = [dP_dtheta, dP_da1,  dP_db1, T2, T3]
vector_names = ['∂P/∂theta',  '∂P/∂a1',  '∂P/∂b1', 'T2', 'T3']

print("=" * 60)
print("验证5个向量的线性无关性")
print("=" * 60)

# 构建系数矩阵（每列是一个向量）
matrix = sp.Matrix.hstack(*vectors)
print(f"系数矩阵维度: {matrix.shape} (8×5)")

# 计算矩阵的秩
rank = matrix.rank()
print(f"矩阵的秩: {rank}")

if rank == 5:
    print("✓ 所有5个向量线性无关")
else:
    print(f"✗ 向量线性相关，实际最大无关组大小为: {rank}")

print("\n" + "=" * 60)
print("检查各种子集的线性无关性")
print("=" * 60)


print("\n" + "=" * 60)
print("符号化验证（使用随机测试点）")
print("=" * 60)



# 选择一个典型测试点
typical_point = {
    theta: sp.pi / 6,
    a0: 0,
    a1: 1.42744876,
    b0: 0.71372438,
    b1: 2.28452071
}

numeric_matrix_typical = matrix.subs(typical_point).evalf()
try:
    # 转换为numpy数组计算条件数
    np_matrix = np.array(numeric_matrix_typical.tolist()).astype(float)
    cond_number = np.linalg.cond(np_matrix)
    print(f"在典型点 {typical_point} 的条件数: {cond_number:.6e}")

    if cond_number > 1e10:
        print("条件数很大，数值计算可能不稳定")
    else:
        print("条件数合理，数值计算稳定")

except Exception as e:
    print(f"条件数计算错误: {e}")