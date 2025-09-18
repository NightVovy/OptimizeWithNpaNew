import sympy as sp

# 定义符号变量
theta, a0, a1, b0, b1 = sp.symbols('theta a0 a1 b0 b1')

# 定义5个向量
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
    0,
    0,
    0,
    -sp.sin(a0) * sp.cos(b0) + sp.sin(2 * theta) * sp.cos(a0) * sp.sin(b0),
    -sp.sin(a0) * sp.cos(b1) + sp.sin(2 * theta) * sp.cos(a0) * sp.sin(b1),
    0,
    0
]

pd_b0 = [
    0,
    0,
    -sp.cos(2 * theta) * sp.sin(b0),
    0,
    -sp.cos(a0) * sp.sin(b0) + sp.sin(2 * theta) * sp.sin(a0) * sp.cos(b0),
    0,
    -sp.cos(a1) * sp.sin(b0) + sp.sin(2 * theta) * sp.sin(a1) * sp.cos(b0),
    0
]

# 创建矩阵A，每一行是一个向量
A = sp.Matrix([T1, T2, T3, pd_a0, pd_b0])

# 计算矩阵的行简化阶梯形式
rref_matrix, pivots = A.rref()

# 确定自由变量的索引
num_vars = A.shape[1]
pivot_cols = set(pivots)
free_vars = [i for i in range(num_vars) if i not in pivot_cols]

# 计算解空间的基
basis = []
for free_var in free_vars:
    # 创建一个解向量，初始化为0（使用标量0而不是矩阵）
    solution = [sp.S(0) for _ in range(num_vars)]  # 修正：使用sp.S(0)代替sp.zeros(1, 1)
    # 自由变量设为1
    solution[free_var] = sp.S(1)  # 修正：使用sp.S(1)确保类型一致
    # 计算主变量的值
    for i in range(len(pivots)):
        row = rref_matrix[i, :]
        pivot_col = pivots[i]
        # 计算主变量的值
        val = -sum(row[j] * solution[j] for j in range(num_vars) if j != pivot_col)
        solution[pivot_col] = val
    # 将解向量添加到基中
    basis.append(sp.Matrix(solution))

# 输出结果
print("解空间的维度:", len(basis))
print("\n解空间的基:")
for i, b in enumerate(basis):
    print(f"\n基向量 {i + 1}:")
    for j in range(8):
        print(f"β_{j + 1} = {sp.simplify(b[j])}")

# 验证基向量是否与原向量正交
print("\n验证正交性（应为0）:")
for i, b in enumerate(basis):
    print(f"\n基向量 {i + 1}:")
    for vec, name in zip([T1, T2, T3, pd_a0, pd_b0], ["T1", "T2", "T3", "pd_a0", "pd_b0"]):
        dot_product = sp.simplify(sp.Matrix(vec).dot(b))
        print(f"与{name}的点积: {dot_product}")
