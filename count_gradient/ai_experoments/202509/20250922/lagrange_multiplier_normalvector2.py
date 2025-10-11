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
    但只使用3个方程（去掉(-1,-1)组合）
    """
    # 4种s,t组合（完整集合）
    all_combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    # 只使用3个组合（去掉(-1,-1)）
    used_combinations = [(1, 1), (1, -1), (-1, 1)]

    gradients = []
    for s, t in all_combinations:
        grad = constraint_gradient(p, s, t)
        gradients.append(grad)

    # 计算加权和的法向量（只使用3个梯度）
    n_vector = np.zeros(8)
    for i, (s, t) in enumerate(all_combinations):
        # 如果是(-1,-1)组合（索引为3），则跳过
        if (s, t) == (-1, -1):
            continue

        # 调整lambda索引映射
        if i < 3:  # (1,1)、(1,-1)、(-1,1) 保持原索引
            lambda_idx = i
        # 注意：(-1,-1)已经被跳过，所以不需要处理索引为3的情况

        n_vector += lambdas[lambda_idx] * gradients[i]

    return n_vector, gradients, used_combinations


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

# 在这里设置lambda值，现在只有3个值（对应3个方程）
lambdas = [0.33, 0.33, 0.34]  # 调整为3个值

# 计算法向量和各个梯度
normal_vector, all_gradients, used_combinations = compute_normal_vector(p, lambdas)

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

# 输出当前lambda组合值（现在只有3个）
print(f"当前lambda组合值: λ1={lambdas[0]}, λ2={lambdas[1]}, λ3={lambdas[2]}")
print(f"使用的s,t组合: {used_combinations}")
print(f"跳过的组合: (-1, -1)")
print()

# 输出所有约束方程的梯度（包括未使用的）
all_combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
for i, (s, t) in enumerate(all_combinations):
    is_used = (s, t) in used_combinations
    status = "使用" if is_used else "未使用"
    print(f"约束方程 f{i + 1} (s={s}, t={t}) 的梯度 [{status}]:")
    print([f"{x:.8f}" for x in all_gradients[i]])
    print()

print("法向量 n (基于3个方程计算，跳过(-1,-1)组合):")
print([f"{x:.8f}" for x in normal_vector])
print()

# 验证约束方程在给定点p的值（应该接近0）
print("约束方程在点p的值（应该接近0）:")
for i, (s, t) in enumerate(all_combinations):
    constraint_value = constraint_equation(p, s, t)
    is_used = (s, t) in used_combinations
    status = "使用" if is_used else "未使用"
    print(f"f{i + 1} (s={s}, t={t}) = {constraint_value:.10f} [{status}]")

# 额外信息：显示各个组合对应的索引
print("\n组合索引对应关系:")
print("索引0: (s=1, t=1)   - f1")
print("索引1: (s=1, t=-1)  - f2")
print("索引2: (s=-1, t=1)  - f3")
print("索引3: (s=-1, t=-1) - f4 (被跳过)")