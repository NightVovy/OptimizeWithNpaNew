import numpy as np
from math import asin, pi

# 尝试导入 tilted-alpha-correlation 中的函数
# 确保 tilted_alpha_correlation.py 在同一目录下
try:
    from tilted_alpha_correlation import get_tilted_expectations
except ImportError:
    print("警告: 未找到 tilted_alpha_correlation.py，请确保文件存在。")


    # 为了防止运行报错，这里定义一个临时的占位函数
    def get_tilted_expectations(alpha):
        raise NotImplementedError("请确保 tilted_alpha_correlation.py 文件与此脚本在同一目录")


# ==========================================
# 第一部分：数学定义 (基于 Theorem 1 修改版)
# ==========================================

def compute_arcsin_term(Ax, alpha, By, s_or_t, Exy):
    """
    辅助函数：计算单个 arcsin 项
    公式: arcsin[(Exy + s*By) / (1 + s*Ax)]
    """
    numerator = Exy + s_or_t * By
    denominator = 1 + s_or_t * Ax

    # 数值稳定性处理：防止浮点误差导致输入超出 [-1, 1]
    val = numerator / denominator
    val = np.clip(val, -1.0, 1.0)
    return asin(val)


def constraint_equation(p, s, t):
    """
    约束方程定义。
    公式: term1 + term2 + term3 - term4 = π
    """
    # 解包点 P 的坐标 (A0, A1, B0, B1, E00, E01, E10, E11)
    A0, A1, B0, B1, E00, E01, E10, E11 = p

    # 对应 E00 (A0, B0)
    term1 = compute_arcsin_term(A0, s, B0, s, E00)
    # 对应 E10 (A1, B0)
    term2 = compute_arcsin_term(A1, t, B0, t, E10)
    # 对应 E01 (A0, B1)
    term3 = compute_arcsin_term(A0, s, B1, s, E01)
    # 对应 E11 (A1, B1) -> 这里前面放负号
    term4 = compute_arcsin_term(A1, t, B1, t, E11)

    return term1 + term2 + term3 - term4 - pi


def get_gradient_at_point(p, s, t, h=1e-8):
    """
    利用有限差分法计算方程在点 p 处的梯度 (法向量)
    """
    grad = np.zeros(8)
    base_val = constraint_equation(p, s, t)

    for i in range(8):
        p_perturb = np.array(p, dtype=float)
        p_perturb[i] += h
        new_val = constraint_equation(p_perturb, s, t)
        grad[i] = (new_val - base_val) / h

    return grad


# ==========================================
# 第二部分：线性无关性分析工具
# ==========================================

def analyze_linear_independence(p_coords):
    print(f"\n{'=' * 50}")
    print(" 线性无关性验证报告 ")
    print(f"{'=' * 50}")

    # 1. 打印输入的点
    print("输入点 P 的坐标:")
    labels = ["A0", "A1", "B0", "B1", "E00", "E01", "E10", "E11"]
    for l, v in zip(labels, p_coords):
        print(f"  {l}: {v:.6f}")

    # 2. 计算 4 个 (s, t) 组合的梯度
    combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    grads = []

    print("\n[1] 计算 4 个约束方程的梯度向量:")
    for i, (s, t) in enumerate(combinations):
        resid = constraint_equation(p_coords, s, t)
        status = "满足" if abs(resid) < 1e-4 else "不满足(偏离边界)"
        g = get_gradient_at_point(p_coords, s, t)
        grads.append(g)
        print(f"  方程 {i + 1} (s={s:2d}, t={t:2d}): 残差={resid:.2e} [{status}]")

    grad_matrix = np.array(grads)  # 4x8 矩阵

    # 3. 奇异值分解分析
    print("\n[2] 梯度矩阵奇异值分析 (Singular Values):")
    U, S, Vh = np.linalg.svd(grad_matrix)
    print(f"  奇异值: {S}")

    # 判断有效秩（物理阈值）
    # 注意：在梯度计算中，1e-5 是一个比较保守的阈值，视 h 的大小而定
    effective_rank = np.sum(S > 1e-4)
    numpy_rank = np.linalg.matrix_rank(grad_matrix)

    print(f"  Numpy默认秩: {numpy_rank}")
    print(f"  有效秩 (Singular Value > 1e-4): {effective_rank}")

    # =========================================================
    # 新增功能：Rank=3 时的依赖关系具体分析
    # =========================================================
    if effective_rank == 3:
        print("\n  结论: 有效秩为 3，存在线性相关性。")
        print("  正在分析具体依赖关系...")
        print("-" * 30)

        # 我们尝试将每一个方程表示为其他三个方程的线性组合
        # 形式: Eq_target = c1*Eq_a + c2*Eq_b + c3*Eq_c

        found_dependency = False
        indices = [0, 1, 2, 3]

        for target_idx in indices:
            # 选出另外三个作为基底
            basis_indices = [i for i in indices if i != target_idx]

            # 构建 A (3x8) 和 b (1x8)
            # 注意：lstsq 求解 Ax = b，其中 x 是系数。
            # 这里我们要解: c1*g_a + c2*g_b + c3*g_c = g_target
            # 转置后: [g_a.T, g_b.T, g_c.T] * [c1, c2, c3].T = g_target.T

            basis_matrix = grad_matrix[basis_indices].T  # 形状 (8, 3)
            target_vector = grad_matrix[target_idx]  # 形状 (8,)

            # 最小二乘求解
            coeffs, residuals, rank_of_basis, s_of_basis = np.linalg.lstsq(basis_matrix, target_vector, rcond=None)

            # 计算重构误差
            reconstructed = np.dot(basis_matrix, coeffs)
            diff = np.linalg.norm(reconstructed - target_vector)

            # 如果误差极小，说明找到了依赖关系
            if diff < 1e-5:
                found_dependency = True
                print(f"  [发现关系] 方程 {target_idx + 1} 可以被其他三个线性表示:")
                print(f"    Eq {target_idx + 1} ≈ ", end="")

                terms = []
                for b_idx, coeff in zip(basis_indices, coeffs):
                    terms.append(f"{coeff:+.4f} * Eq {b_idx + 1}")
                print(" ".join(terms))
                print(f"    (重构误差: {diff:.2e})")

                # 通常找到一个关系就可以说明问题了，因为它们互相关联
                # 但为了全面，我们继续循环，展示所有可能的关系

        if not found_dependency:
            print("  [异常] 判定为秩3，但未能用最小二乘法找到精确的线性组合，可能数值噪声过大。")

    elif effective_rank == 4:
        print("\n  结论: 4 个向量线性无关 (Rank=4)。")
    else:
        print(f"\n  结论: 秩为 {effective_rank}，简并度更高。")


# ==========================================
# 第三部分：用户输入区域
# ==========================================

if __name__ == "__main__":
    # 设定 alpha
    alpha = 0.5
    print(f"正在计算 alpha={alpha} 时的 Tilted-CHSH 量子点...")

    # 从 titled_alpha_correlation 获取 8 个期望值
    p_input = get_tilted_expectations(alpha)

    # 运行分析
    analyze_linear_independence(p_input)