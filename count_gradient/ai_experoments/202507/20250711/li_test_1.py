import sympy as sp
from sympy import sin, cos, asin, atanh


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
    equation = term1 + term2 + term3 + term4
    return term1, term2, term3, term4, equation


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

    # 输出参数值
    print("\n参数表达式值：")
    print(f"A0 = {A0.subs(subs_dict).evalf()}")
    print(f"A1 = {A1.subs(subs_dict).evalf()}")
    print(f"B0 = {B0.subs(subs_dict).evalf()}")
    print(f"B1 = {B1.subs(subs_dict).evalf()}")
    print(f"E00 = {E00.subs(subs_dict).evalf()}")
    print(f"E01 = {E01.subs(subs_dict).evalf()}")
    print(f"E10 = {E10.subs(subs_dict).evalf()}")
    print(f"E11 = {E11.subs(subs_dict).evalf()}")

    # 输出每个组合的terms值
    print("\n各项的值：")
    for i, (s_val, t_val) in enumerate(combinations, 1):
        term1, term2, term3, term4, _ = original_equation(s_val, t_val)
        print(f"\n组合 {i} (s={s_val}, t={t_val}):")
        print(f"Term1: {term1.subs(subs_dict).evalf()}")
        print(f"Term2: {term2.subs(subs_dict).evalf()}")
        print(f"Term3: {term3.subs(subs_dict).evalf()}")
        print(f"Term4: {term4.subs(subs_dict).evalf()}")

    # 计算四个方程的数值
    numerical_equations = [eq[-1].subs(subs_dict).evalf() for eq in equations]

    # 输出方程结果
    print("\n方程结果：")
    for i, eq in enumerate(numerical_equations, 1):
        print(f"方程 {i}: {eq}")

    # 检查线性无关性
    # 构造数值矩阵（考虑浮点精度）
    try:
        # 尝试将方程结果转换为浮点数
        float_eqs = [float(eq) for eq in numerical_equations]
        # 构建矩阵（每个方程作为一个行向量）
        matrix = sp.Matrix([[eq] for eq in float_eqs])
        # 计算矩阵的秩
        rank = matrix.rank()
        if rank == 4:
            print("\n四个方程数值线性无关")
        else:
            print(f"\n方程线性相关，秩为 {rank}")
    except:
        print("\n无法转换为数值矩阵进行线性无关性检测")


# 选择参数值
theta_val = sp.pi / 6
a0_val = 0
a1_val = sp.pi / 2
b0_val = atanh(sin(2 * theta_val))
b1_val = - atanh(sin(2 * theta_val))

print("\n参数值：")
print(f"theta = {theta_val}, a0 = {a0_val}, a1 = {a1_val}, b0 = {b0_val}, b1 = {b1_val}")
numerical_check(theta_val, a0_val, a1_val, b0_val, b1_val)