import sympy as sp
import numpy as np

# 定义符号变量
theta, a0, a1, b0, b1 = sp.symbols('theta a0 a1 b0 b1', real=True)
s_val, t_val = sp.symbols('s t', real=True)  # s和t的取值

# 定义中间变量
c2t = sp.cos(2 * theta)  # cos(2θ)
s2t = sp.sin(2 * theta)  # sin(2θ)

# 用5个变量表示8个变量
A0 = c2t * sp.cos(a0)
A1 = c2t * sp.cos(a1)
B0 = c2t * sp.cos(b0)
B1 = c2t * sp.cos(b1)

E00 = sp.cos(a0) * sp.cos(b0) + s2t * sp.sin(a0) * sp.sin(b0)
E01 = sp.cos(a0) * sp.cos(b1) + s2t * sp.sin(a0) * sp.sin(b1)
E10 = sp.cos(a1) * sp.cos(b0) + s2t * sp.sin(a1) * sp.sin(b0)
E11 = sp.cos(a1) * sp.cos(b1) + s2t * sp.sin(a1) * sp.sin(b1)


# 定义单个项的函数
def arcsin_term(E, B, A, s):
    """计算单个arcsin项"""
    numerator = E + s * B
    denominator = 1 + s * A
    u = numerator / denominator
    return sp.asin(u)


# 定义完整的方程
def F_equation(s, t):
    """完整的方程 F(s,t)"""
    term1 = arcsin_term(E00, B0, A0, s)
    term2 = arcsin_term(E10, B0, A1, t)
    term3 = arcsin_term(E01, B1, A0, s)
    term4 = arcsin_term(E11, B1, A1, t)
    return term1 + term2 - term3 + term4


# 定义4种(s,t)组合
s_t_combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

# 存储结果
equations = []
gradients = []

print("=" * 50)
print("4个方程的符号表达式:")
print("=" * 50)

for i, (s, t) in enumerate(s_t_combinations):
    print(f"\n方程 {i + 1} (s={s}, t={t}):")

    # 计算方程
    F = F_equation(s, t)
    equations.append(F)

    # 显示方程
    print(f"F_{i + 1} = {F}")

    # 计算梯度（法向量）
    print(f"\n计算梯度...")
    grad = []
    variables = [theta, a0, a1, b0, b1]

    for var in variables:
        derivative = sp.diff(F, var)
        grad.append(derivative)
        print(f"∂F_{i + 1}/∂{var} = {derivative}")

    gradients.append(grad)
    print("-" * 30)

print("=" * 50)
print("结果汇总:")
print("=" * 50)

# 显示所有方程和梯度
for i, (s, t) in enumerate(s_t_combinations):
    print(f"\n方程 {i + 1} (s={s}, t={t}):")
    print(f"F_{i + 1} = {equations[i]}")
    print(f"梯度向量 ∇F_{i + 1} = ")
    for j, var in enumerate(['θ', 'a0', 'a1', 'b0', 'b1']):
        print(f"  ∂/∂{var}: {gradients[i][j]}")

# 可选：简化表达式（可能会很耗时）
print("\n正在尝试简化表达式...")
simplified_gradients = []
for i, grad in enumerate(gradients):
    simplified = [sp.simplify(expr) for expr in grad]
    simplified_gradients.append(simplified)
    print(f"\n方程 {i + 1} 的简化梯度:")
    for j, var in enumerate(['θ', 'a0', 'a1', 'b0', 'b1']):
        print(f"  ∂/∂{var}: {simplified[j]}")

print("=" * 50)
print("推导完成!")
print("=" * 50)