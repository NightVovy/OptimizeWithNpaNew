import sympy as sp

# 定义符号变量
s, t = sp.symbols('s t')
A0, A1, B0, B1, E00, E01, E10, E11 = sp.symbols('A0 A1 B0 B1 E00 E01 E10 E11')

# 定义原始方程
def original_equation(s_val, t_val, A0_val, A1_val, B0_val, B1_val, E00_val, E01_val, E10_val, E11_val):
    term1 = sp.asin((E00 + s_val * B0) / (1 + s_val * A0_val))
    term2 = sp.asin((E01 + s_val * B1) / (1 + s_val * A0_val))
    term3 = sp.asin((E10 + t_val * B0) / (1 + t_val * A1_val))
    term4 = -sp.asin((E11 + t_val * B1) / (1 + t_val * A1_val))
    equation = term1 + term2 + term3 + term4 - sp.pi
    return equation

# 四种 (s, t) 组合
combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

# 情况 1: 只有 A0 或 A1 中有一个为 0
def case1(A0_zero=True):
    params = {
        A0: 0 if A0_zero else sp.Symbol('A0'),
        A1: sp.Symbol('A1') if A0_zero else 0,
        B0: sp.Symbol('B0'),
        B1: sp.Symbol('B1'),
        E00: sp.Symbol('E00'),
        E01: sp.Symbol('E01'),
        E10: sp.Symbol('E10'),
        E11: sp.Symbol('E11')
    }
    equations = []
    for s_val, t_val in combinations:
        eq = original_equation(s_val, t_val, params[A0], params[A1], params[B0], params[B1],
                              params[E00], params[E01], params[E10], params[E11])
        equations.append(eq)
    return equations

# 情况 2: 所有变量均不为 0
def case2():
    params = {
        A0: sp.Symbol('A0'),
        A1: sp.Symbol('A1'),
        B0: sp.Symbol('B0'),
        B1: sp.Symbol('B1'),
        E00: sp.Symbol('E00'),
        E01: sp.Symbol('E01'),
        E10: sp.Symbol('E10'),
        E11: sp.Symbol('E11')
    }
    equations = []
    for s_val, t_val in combinations:
        eq = original_equation(s_val, t_val, params[A0], params[A1], params[B0], params[B1],
                              params[E00], params[E01], params[E10], params[E11])
        equations.append(eq)
    return equations

# 检查线性无关性
def check_linear_independence(equations):
    # 尝试符号化求解
    unique_eqs = []
    for eq in equations:
        simplified = sp.simplify(eq)
        is_unique = True
        for ueq in unique_eqs:
            if sp.simplify(simplified - ueq) == 0:
                is_unique = False
                break
        if is_unique:
            unique_eqs.append(simplified)
    return len(unique_eqs)

# 情况 1: A0 = 0, A1 ≠ 0
print("情况 1: A0 = 0, A1 ≠ 0")
equations_case1_A0_zero = case1(A0_zero=True)
num_independent_case1_A0_zero = check_linear_independence(equations_case1_A0_zero)
print(f"独立方程数量: {num_independent_case1_A0_zero}")
for i, eq in enumerate(equations_case1_A0_zero, 1):
    print(f"方程 {i}: {eq}")

# 情况 1: A1 = 0, A0 ≠ 0
print("\n情况 1: A1 = 0, A0 ≠ 0")
equations_case1_A1_zero = case1(A0_zero=False)
num_independent_case1_A1_zero = check_linear_independence(equations_case1_A1_zero)
print(f"独立方程数量: {num_independent_case1_A1_zero}")
for i, eq in enumerate(equations_case1_A1_zero, 1):
    print(f"方程 {i}: {eq}")

# 情况 2: 所有变量均不为 0
print("\n情况 2: 所有变量均不为 0")
equations_case2 = case2()
num_independent_case2 = check_linear_independence(equations_case2)
print(f"独立方程数量: {num_independent_case2}")
for i, eq in enumerate(equations_case2, 1):
    print(f"方程 {i}: {eq}")