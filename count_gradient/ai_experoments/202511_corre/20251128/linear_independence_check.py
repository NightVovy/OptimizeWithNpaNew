import numpy as np
from math import asin, pi


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
    根据你的要求，修改符号为：前三项为正，最后一项(A1, B1)为负。
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

    # 这里的正负号顺序决定了方程的形式
    # 修改版: + + + -
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
        # 验证方程是否满足 (残差应接近 0)
        resid = constraint_equation(p_coords, s, t)
        status = "满足" if abs(resid) < 1e-4 else "不满足(偏离边界)"

        g = get_gradient_at_point(p_coords, s, t)
        grads.append(g)

        print(f"  方程 {i + 1} (s={s:2d}, t={t:2d}): 残差={resid:.2e} [{status}]")
        # 简单展示梯度中 E 部分的符号 (E00, E01, E10, E11)
        e_signs = np.sign(g[4:])
        print(f"    -> 梯度关联项符号: {e_signs}")

    grad_matrix = np.array(grads)  # 4x8 矩阵

    # 3. 计算矩阵的秩 (Rank)
    rank = np.linalg.matrix_rank(grad_matrix)
    print(f"\n[2] 梯度矩阵的秩 (Rank): {rank} / 4")

    if rank == 4:
        print("  结论: 4 个向量【线性无关】。")
        print("  这意味着没有任何一个方程可以由其他三个方程线性表示。")
        print("  (它们描述了高维空间中相交于该顶点的 4 个不同超曲面)")
    else:
        print(f"  结论: 存在线性相关，只有 {rank} 个独立方向。")
        # 尝试找出相关性：尝试用前3个表示第4个
        # g4 = a*g1 + b*g2 + c*g3
        A = grad_matrix[:3].T
        b = grad_matrix[3]
        x, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)

        diff = np.linalg.norm(np.dot(A, x) - b)
        if diff < 1e-6:
            print("\n[3] 线性关系识别:")
            print(f"  方程 4 可以被表示为前 3 个的组合:")
            print(f"  g4 ≈ {x[0]:.3f} * g1 + {x[1]:.3f} * g2 + {x[2]:.3f} * g3")
        else:
            print("\n[3] 无法简单用前3个表示第4个 (可能依赖关系更复杂)")

    # 4. 物理洞察：梯度的和是否构成 CHSH?
    # 标准 CHSH 的法向量 (无边缘项，E项系数为 1,1,1,-1)
    sum_grads = np.sum(grad_matrix, axis=0)
    # 归一化以便比较
    if np.linalg.norm(sum_grads) > 1e-9:
        sum_grads_norm = sum_grads / np.max(np.abs(sum_grads))
    else:
        sum_grads_norm = sum_grads

    print("\n[4] 物理检查: 4 个梯度的直接求和")
    print("  (如果边缘项 A/B 为 0，且 E 项非 0，说明它们合成了 Bell 不等式)")
    print(f"  求和向量: {np.round(sum_grads_norm, 4)}")

    if np.allclose(sum_grads_norm[:4], 0) and not np.allclose(sum_grads_norm[4:], 0):
        print("  ==> 完美! 4 个方程的梯度之和正好消去了局部边缘项 (A0..B1)。")
        print("  ==> 剩下的部分正是 CHSH 不等式的系数形式。")


# ==========================================
# 第三部分：用户输入区域
# ==========================================

if __name__ == "__main__":
    # --- 在这里修改你的输入点 P ---
    # 格式: [A0, A1, B0, B1, E00, E01, E10, E11]

    # 场景 1: 标准 CHSH 最大违背点 P1
    # 边缘项为 0，关联项为 1/sqrt(2) 或 -1/sqrt(2)
    val = 1 / np.sqrt(2)
    p_input = [
        0, 0, 0, 0,  # A0, A1, B0, B1
        val, val, val, -val  # E00, E01, E10, E11 (负号在最后一项)
    ]

    # 运行分析
    analyze_linear_independence(p_input)