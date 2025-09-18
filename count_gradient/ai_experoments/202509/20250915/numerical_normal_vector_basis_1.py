import sympy as sp
import numpy as np

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

# 创建符号矩阵A
A_symbolic = sp.Matrix([T1, T2, T3, pd_a0, pd_b0])

# 代入数值
theta_val = sp.pi / 6
a0_val = 0
a1_val = 1.42744876
b0_val = 0.71372438
b1_val = 2.28452071

# 数值替换字典
subs_dict = {
    theta: theta_val,
    a0: a0_val,
    a1: a1_val,
    b0: b0_val,
    b1: b1_val
}

# 创建数值矩阵A
A_numeric = A_symbolic.subs(subs_dict)

# 将符号表达式转换为数值
A_numeric = A_numeric.applyfunc(lambda x: float(x.evalf()) if x.is_number else float(x))

print("数值矩阵A:")
sp.pprint(A_numeric)

# 计算矩阵的行简化阶梯形式
try:
    rref_matrix, pivots = A_numeric.rref()
    print("\n行简化阶梯形式:")
    sp.pprint(rref_matrix)
    print("主元列:", pivots)

    # 确定自由变量的索引
    num_vars = A_numeric.shape[1]
    pivot_cols = set(pivots)
    free_vars = [i for i in range(num_vars) if i not in pivot_cols]

    print(f"\n自由变量索引: {free_vars}")
    print(f"解空间的维度: {len(free_vars)}")

    # 计算解空间的基
    basis = []
    for free_var in free_vars:
        # 创建一个解向量，初始化为0
        solution = [0.0 for _ in range(num_vars)]
        # 自由变量设为1
        solution[free_var] = 1.0
        # 计算主变量的值
        for i in range(len(pivots)):
            row = rref_matrix.row(i)
            pivot_col = pivots[i]
            # 计算主变量的值
            val = -sum(float(row[j]) * solution[j] for j in range(num_vars) if j != pivot_col)
            solution[pivot_col] = val
        # 将解向量添加到基中
        basis.append(sp.Matrix(solution))

    # 输出结果
    print("\n解空间的基:")
    for i, b in enumerate(basis):
        print(f"\n基向量 {i + 1}:")
        for j in range(8):
            print(f"β_{j + 1} = {b[j]:.6f}")

    # 验证基向量是否与原向量正交
    print("\n验证正交性（应为0）:")
    vectors_numeric = [
        A_numeric.row(0),  # T1
        A_numeric.row(1),  # T2
        A_numeric.row(2),  # T3
        A_numeric.row(3),  # pd_a0
        A_numeric.row(4)  # pd_b0
    ]

    for i, b in enumerate(basis):
        print(f"\n基向量 {i + 1}:")
        for j, vec in enumerate(vectors_numeric):
            dot_product = float(vec.dot(b))
            print(f"与向量{j + 1}的点积: {dot_product:.10f}")

except Exception as e:
    print(f"计算过程中出现错误: {e}")
    print("检查矩阵A:")
    sp.pprint(A_numeric)

    # 检查是否有除0错误
    print("\n检查矩阵元素:")
    for i in range(A_numeric.shape[0]):
        for j in range(A_numeric.shape[1]):
            element = A_numeric[i, j]
            if abs(element) < 1e-10:  # 接近0的元素
                print(f"A[{i},{j}] = {element:.10f} (接近0)")