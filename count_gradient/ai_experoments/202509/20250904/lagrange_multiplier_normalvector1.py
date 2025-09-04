import numpy as np
from math import asin, pi


def compute_arcsin_term(Ax, alpha, By, s_or_t, Exy):
    """
    计算 arcsin[(Exy + alpha * By) / (1 + alpha * Ax)]
    其中 s_or_t 是 s 或 t，取值为 1 或 -1
    """
    numerator = Exy + s_or_t * By
    denominator = 1 + s_or_t * Ax
    return asin(numerator / denominator)


def constraint_equation(p, s, t):
    """
    计算约束方程: arcsin[A0 s B0] + arcsin[A1 t B0] - arcsin[A0 s B1] + arcsin[A1 t B1] = π
    """
    A0, A1, B0, B1, E00, E01, E10, E11 = p

    term1 = compute_arcsin_term(A0, s, B0, s, E00)  # arcsin[A0 s B0]
    term2 = compute_arcsin_term(A1, t, B0, t, E10)  # arcsin[A1 t B0]
    term3 = compute_arcsin_term(A0, s, B1, s, E01)  # arcsin[A0 s B1]
    term4 = compute_arcsin_term(A1, t, B1, t, E11)  # arcsin[A1 t B1]

    return term1 + term2 - term3 + term4 - pi


def constraint_gradient(p, s, t, h=1e-8):
    """
    计算约束方程在点p处的梯度（数值微分）
    """
    grad = np.zeros(8)
    original_constraint = constraint_equation(p, s, t)

    for i in range(8):
        p_perturbed = np.array(p, dtype=float)
        p_perturbed[i] += h
        perturbed_constraint = constraint_equation(p_perturbed, s, t)
        grad[i] = (perturbed_constraint - original_constraint) / h

    return grad


def compute_normal_vector(p, lambdas):
    """
    计算法向量 n = λ1·∇f1 + λ2·∇f2 + λ3·∇f3 + λ4·∇f4
    """
    # 4种s,t组合
    combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    gradients = []
    for s, t in combinations:
        grad = constraint_gradient(p, s, t)
        gradients.append(grad)

    # 计算加权和的法向量
    n_vector = np.zeros(8)
    for i in range(4):
        n_vector += lambdas[i] * gradients[i]

    return n_vector, gradients


# 给定的点p
p = np.array([
    0.37796447,  # A0
    0.37796447,  # A1
    0.5,  # B0
    0.0,  # B1
    0.75592895,  # E00
    -0.56694671,  # E01
    0.75592895,  # E10
    0.56694671  # E11
])

# 在这里设置lambda值，方便修改
lambdas = [1, 1, 1, 1]

# 计算法向量和各个梯度
normal_vector, all_gradients = compute_normal_vector(p, lambdas)

print("点 p 的坐标:")
print(f"A0 = {p[0]:.8f}")
print(f"A1 = {p[1]:.8f}")
print(f"B0 = {p[2]:.8f}")
print(f"B1 = {p[3]:.8f}")
print(f"E00 = {p[4]:.8f}")
print(f"E01 = {p[5]:.8f}")
print(f"E10 = {p[6]:.8f}")
print(f"E11 = {p[7]:.8f}")
print()

# 输出当前lambda组合值
print(f"当前lambda组合值: λ1={lambdas[0]}, λ2={lambdas[1]}, λ3={lambdas[2]}, λ4={lambdas[3]}")
print()

# 输出各个约束方程的梯度
combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
for i, (s, t) in enumerate(combinations):
    print(f"约束方程 f{i + 1} (s={s}, t={t}) 的梯度:")
    print([f"{x:.8f}" for x in all_gradients[i]])
    print()

print("法向量 n:")
print([f"{x:.8f}" for x in normal_vector])
print()

# 验证约束方程在给定点p的值（应该接近0）
print("约束方程在点p的值（应该接近0）:")
for i, (s, t) in enumerate(combinations):
    constraint_value = constraint_equation(p, s, t)
    print(f"f{i + 1} (s={s}, t={t}) = {constraint_value:.10f}")