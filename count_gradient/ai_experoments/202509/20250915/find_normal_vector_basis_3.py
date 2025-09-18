import sympy as sp

# 定义符号变量
theta, a0, a1, b0, b1 = sp.symbols('theta a0 a1 b0 b1')
u, v, w = sp.symbols('u v w')  # 添加系数符号

# 定义具体数值
theta_val = sp.pi / 6
a0_val = 0
a1_val = 1.42744876
b0_val = 0.71372438
b1_val = 2.28452071

var_dict = {
    theta: theta_val,
    a0: a0_val,
    a1: a1_val,
    b0: b0_val,
    b1: b1_val
}


# 定义5个向量的生成函数
def create_vectors(theta, a0, a1, b0, b1):
    T1 = [  # theta
        -2 * sp.sin(2 * theta) * sp.cos(a0),
        -2 * sp.sin(2 * theta) * sp.cos(a1),
        -2 * sp.sin(2 * theta) * sp.cos(b0),
        -2 * sp.sin(2 * theta) * sp.cos(b1),
        2 * sp.sin(a0) * sp.sin(b0) * sp.cos(2 * theta),
        2 * sp.sin(a0) * sp.sin(b1) * sp.cos(2 * theta),
        2 * sp.sin(a1) * sp.sin(b0) * sp.cos(2 * theta),
        2 * sp.sin(a1) * sp.sin(b1) * sp.cos(2 * theta)
    ]

    T2 = [  # |01>
        sp.sin(a0) * sp.sin(theta),
        sp.sin(a1) * sp.sin(theta),
        sp.sin(b0) * sp.sin(theta),
        sp.sin(b1) * sp.sin(theta),
        sp.cos(a0) * sp.sin(b0) * sp.cos(theta) - sp.sin(a0) * sp.cos(b0) * sp.sin(theta),
        sp.cos(a0) * sp.sin(b1) * sp.cos(theta) - sp.sin(a0) * sp.cos(b1) * sp.sin(theta),
        sp.cos(a1) * sp.sin(b0) * sp.cos(theta) - sp.sin(a1) * sp.cos(b0) * sp.sin(theta),
        sp.cos(a1) * sp.sin(b1) * sp.cos(theta) - sp.sin(a1) * sp.cos(b1) * sp.sin(theta)
    ]

    T3 = [  # |10>
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

    return T1, T2, T3, pd_a0, pd_b0


# 先代入具体数值得到向量
T1_num, T2_num, T3_num, pd_a0_num, pd_b0_num = create_vectors(
    theta_val, a0_val, a1_val, b0_val, b1_val
)


# 检查表达式是否有除0风险的函数
def check_division_by_zero(expr):
    """检查表达式是否会导致除0错误"""
    try:
        # 对于已经代入数值的表达式，直接检查是否有限
        if not sp.N(expr).is_finite:
            return True, f"表达式 {expr} 计算结果为无穷大，可能存在除0"
    except Exception as e:
        return True, f"表达式 {expr} 计算时出现错误: {str(e)}"
    return False, ""


print("=" * 50)
print("代入具体数值后的表达式检查:")
print("=" * 50)

# 检查所有向量中的表达式是否有除0风险
vectors = [T1_num, T2_num, T3_num, pd_a0_num, pd_b0_num]
vector_names = ["T1", "T2", "T3", "pd_a0", "pd_b0"]
division_zero_detected = False
error_messages = []

for vec, name in zip(vectors, vector_names):
    print(f"\n检查向量 {name}:")
    for j, expr in enumerate(vec):
        has_division_zero, error_msg = check_division_by_zero(expr)
        if has_division_zero:
            division_zero_detected = True
            error_messages.append(f"{name} 的第 {j + 1} 个元素: {error_msg}")
            print(f"  元素 {j + 1}: 存在问题 - {error_msg}")
        else:
            print(f"  元素 {j + 1}: 安全")

# 如果有除0风险，终止程序
if division_zero_detected:
    print("\n" + "!" * 60)
    print("错误: 检测到除0风险，程序终止!")
    print("!" * 60)
    for msg in error_messages:
        print(f"- {msg}")
    exit(1)

print("\n所有表达式检查通过，无除0风险，继续执行...")

# 创建数值矩阵A，每一行是一个代入数值后的向量
A = sp.Matrix([T1_num, T2_num, T3_num, pd_a0_num, pd_b0_num])

# 计算矩阵的行简化阶梯形式
rref_matrix, pivots = A.rref()

# 确定自由变量的索引
num_vars = A.shape[1]
pivot_cols = set(pivots)
free_vars = [i for i in range(num_vars) if i not in pivot_cols]

# 计算解空间的基
basis = []
for free_var in free_vars:
    # 创建一个解向量，初始化为0
    solution = [sp.S(0) for _ in range(num_vars)]
    # 自由变量设为1
    solution[free_var] = sp.S(1)
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
print("\n" + "=" * 50)
print("解空间的基向量计算结果:")
print("=" * 50)
print("解空间的维度:", len(basis))
print("\n解空间的基:")
for i, b in enumerate(basis):
    print(f"\n基向量 {i + 1}:")
    for j in range(8):
        # 简化并转换为数值形式输出
        val = sp.N(b[j])
        print(f"β_{j + 1} = {val:.6f}")

# 验证基向量是否与原向量正交
print("\n" + "=" * 50)
print("验证正交性（应为接近0的值）:")
print("=" * 50)
for i, b in enumerate(basis):
    print(f"\n基向量 {i + 1}:")
    for vec, name in zip(vectors, vector_names):
        dot_product = sp.N(sp.Matrix(vec).dot(b))
        print(f"与{name}的点积: {dot_product:.10f}")

# 输出解向量表达式
print("\n" + "=" * 50)
print("解向量表达式:")
print("=" * 50)

if len(basis) == 3:
    print("β = u * (基向量1) + v * (基向量2) + w * (基向量3)")
    print("\n其中:")
    for i, b in enumerate(basis):
        print(f"基向量 {i + 1} = {[sp.N(x) for x in b]}")
elif len(basis) > 0:
    print("β = " + " + ".join([f"{coeff} * (基向量{i + 1})" for i, coeff in enumerate([u, v, w][:len(basis)])]))
    print("\n其中:")
    for i, b in enumerate(basis):
        print(f"基向量 {i + 1} = {[sp.N(x) for x in b]}")
else:
    print("无解空间")

print("\n程序执行完成!")
