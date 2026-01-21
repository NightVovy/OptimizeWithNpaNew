import numpy as np
from math import asin, pi
import sys

# 尝试导入同目录下的 correlation 计算模块
try:
    from hardy_correlation import calculate_hardy_correlations
except ImportError:
    print("错误: 未找到 'hardy_correlation.py'。请确保该文件在同一目录下。")
    sys.exit(1)


# --- 1. 基础计算函数 (保持原逻辑) ---

def compute_arcsin_term(Ax, alpha, By, s_or_t, Exy):
    """
    计算 arcsin[(Exy + alpha * By) / (1 + alpha * Ax)]
    """
    numerator = Exy + s_or_t * By
    denominator = 1 + s_or_t * Ax

    # 防止数值误差导致超出 arcsin 定义域 (-1, 1)
    if abs(denominator) < 1e-9:
        val = 0  # 避免除零
    else:
        val = numerator / denominator

    val = np.clip(val, -1.0, 1.0)
    return asin(val)


def constraint_equation(p, s, t, print_terms=False):
    """
    约束方程 (Standard CHSH 形式):
    arcsin[A0 s B0] + arcsin[A1 t B0] + arcsin[A0 s B1] - arcsin[A1 t B1] = π
    符号结构: + + + -
    """
    # p 的顺序: [A0, A1, B0, B1, E00, E01, E10, E11]
    A0, A1, B0, B1, E00, E01, E10, E11 = p

    # 对应 E00: A0, B0
    term1 = compute_arcsin_term(A0, s, B0, s, E00)
    # 对应 E10: A1, B0
    term2 = compute_arcsin_term(A1, t, B0, t, E10)
    # 对应 E01: A0, B1
    term3 = compute_arcsin_term(A0, s, B1, s, E01)
    # 对应 E11: A1, B1
    term4 = compute_arcsin_term(A1, t, B1, t, E11)

    # Standard CHSH form: + + + -
    result = term1 + term2 + term3 - term4 - pi

    if print_terms:
        print(f"\n--- 约束方程细节 (s={s}, t={t}) ---")
        print(f"Term1 (A0B0): {term1:.8f}")
        print(f"Term2 (A1B0): {term2:.8f}")
        print(f"Term3 (A0B1): {term3:.8f}")
        print(f"Term4 (A1B1): {term4:.8f}")
        print(f"Result (Sum - pi): {result:.8e}")

    return result


def constraint_gradient(p, s, t, h=1e-8):
    """数值梯度计算"""
    grad = np.zeros(8)
    original_constraint = constraint_equation(p, s, t)

    for i in range(8):
        p_perturbed = np.array(p, dtype=float)
        p_perturbed[i] += h
        perturbed_constraint = constraint_equation(p_perturbed, s, t)
        grad[i] = (perturbed_constraint - original_constraint) / h

    return grad


def compute_normal_vector(p, lambdas):
    """计算加权法向量"""
    # 完整的 s,t 组合 (定义了量子集合边界的四个相交面)
    all_combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    gradients = []
    n_vector = np.zeros(8)

    # 1. 计算所有 4 个梯度
    for s, t in all_combinations:
        gradients.append(constraint_gradient(p, s, t))

    # 2. 加权求和
    if len(lambdas) != 4:
        raise ValueError("lambdas 列表长度必须为 4")

    print("\n>>> 开始加权计算法向量 <<<")
    for i, (s, t) in enumerate(all_combinations):
        weight = lambdas[i]
        grad = gradients[i]
        n_vector += weight * grad
        # print(f"  组合 (s={s:2d}, t={t:2d}) 权重: {weight:.2f}")

    return n_vector


# --- 2. 主执行逻辑 ---

if __name__ == "__main__":
    # 1. 获取 Hardy Point 数据
    print("正在调用 hardy_correlation 获取数据...")
    # 返回顺序: [A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1]
    hardy_data_list = calculate_hardy_correlations()

    # 转换为 numpy 数组，作为点 p
    p_hardy = np.array(hardy_data_list, dtype=float)

    print("\n>>> Hardy Point 坐标 (P) <<<")
    labels = ["A0", "A1", "B0", "B1", "A0B0", "A0B1", "A1B0", "A1B1"]
    for l, v in zip(labels, p_hardy):
        print(f"{l:<4}: {v:.8f}")

    # 2. 验证该点是否在量子边界上 (代入约束方程)
    print("\n>>> 验证 Hardy Point 是否满足边界方程 (Res -> 0) <<<")
    # Hardy 态通常位于边界上，残差应非常小
    combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for s, t in combinations:
        res = constraint_equation(p_hardy, s, t)
        print(f"  (s={s:2d}, t={t:2d}) 残差: {res:.8e}")

    # 3. 设置 Lagrange 乘子 (权重)
    # 这里的 lambda 决定了我们在切平面上的具体取向
    lambdas = [0.1, 0.4, 0.2, 0.3]

    # 4. 计算法向量
    normal_vec = compute_normal_vector(p_hardy, lambdas)

    # 5. 打印结果
    print("\n>>> 计算得到的 Bell 不等式系数 (法向量) <<<")
    # 修改 1: 输出当前 correlation 组合
    print("当前correlation组合: [A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1]")

    # 归一化以便观察相对大小 (以绝对值最大项为基准)
    max_val = np.max(np.abs(normal_vec))
    if max_val > 1e-9:
        normalized_n = normal_vec / max_val
    else:
        normalized_n = normal_vec

    # 修改 2: 输出当前 lambda 取值
    print(f"当前lambda取值: {lambdas}")
    print(f"原始向量: {np.round(normal_vec, 6)}")
    print(f"归一化后: {np.round(normalized_n, 6)}")

    print("\n>>> 结果分析 <<<")
    print("该法向量定义了一个 Bell 不等式: I · P <= bound")
    print("由于 Hardy Point 的不对称性，此处得到的法向量通常包含非零的单体项系数 (A0, A1...)。")
    print("这与 CHSH 不同，表明检测该点的最优不等式通常是有偏的 (tilted)。")