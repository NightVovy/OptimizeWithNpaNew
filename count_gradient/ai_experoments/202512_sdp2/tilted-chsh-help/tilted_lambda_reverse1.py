import numpy as np
from math import asin, pi
import itertools

# 尝试导入 tilted-alpha-correlation 中的函数
try:
    from tilted_alpha_correlation import get_tilted_expectations
except ImportError:
    print("警告: 未找到 tilted_alpha_correlation.py，请确保文件存在。")


    def get_tilted_expectations(alpha):
        raise NotImplementedError("请确保 tilted_alpha_correlation.py 文件与此脚本在同一目录")


# ==========================================
# 第一部分：复用参考代码中的数学定义
# (源自: normal_vector_tilted_chsh.py)
# ==========================================

def compute_arcsin_term(Ax, alpha, By, s_or_t, Exy):
    """
    辅助函数：计算单个 arcsin 项
    [复用说明] 直接复制自 normal_vector_tilted_chsh.py
    """
    numerator = Exy + s_or_t * By
    denominator = 1 + s_or_t * Ax
    val = numerator / denominator
    val = np.clip(val, -1.0, 1.0)
    return asin(val)


def constraint_equation(p, s, t):
    """
    约束方程定义
    [复用说明] 直接复制自 normal_vector_tilted_chsh.py
    """
    A0, A1, B0, B1, E00, E01, E10, E11 = p
    term1 = compute_arcsin_term(A0, s, B0, s, E00)
    term2 = compute_arcsin_term(A1, t, B0, t, E10)
    term3 = compute_arcsin_term(A0, s, B1, s, E01)
    term4 = compute_arcsin_term(A1, t, B1, t, E11)
    return term1 + term2 + term3 - term4 - pi


def get_gradient_at_point(p, s, t, h=1e-8):
    """
    数值梯度计算
    [复用说明] 直接复制自 normal_vector_tilted_chsh.py
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
# 第二部分：反向推导 Lambda 的核心逻辑 (修改版)
# ==========================================

def solve_lambdas_for_combination(gradients, combination_indices, alpha):
    """
    针对给定的 3 个方程组合，求解最佳 Lambda，并返回完整的 4 元素 Lambda 数组。
    """
    # 1. 构建目标向量 (Target Tilted-CHSH form)
    # 顺序: [A0, A1, B0, B1, E00, E01, E10, E11]
    # 目标系数: alpha*A0 + 1*A0B0 + 1*A0B1 + 1*A1B0 - 1*A1B1
    target_vec = np.array([alpha, 0, 0, 0, 1, 1, 1, -1])

    # 2. 提取选定的梯度向量作为基底矩阵
    # 矩阵 A 的形状为 (8, 3)，每一列是一个梯度向量
    selected_grads = [gradients[i] for i in combination_indices]
    A = np.column_stack(selected_grads)

    # 3. 使用最小二乘法求解 A * lambda_subset = target_vec
    # lambda_subset 只有 3 个元素
    lambda_subset, residuals, rank, s = np.linalg.lstsq(A, target_vec, rcond=None)

    # 4. 构建完整的 4 元素 Lambda 数组
    # 初始化为全 0
    full_lambdas = np.zeros(4)
    # 将解出的 3 个值填入对应的索引位置
    for i, idx in enumerate(combination_indices):
        full_lambdas[idx] = lambda_subset[i]

    # 5. 重构向量以验证误差
    # 使用完整的 4 个梯度和完整的 4 个 lambda 重构
    # gradients 列表包含所有 4 个梯度
    grad_matrix_full = np.column_stack(gradients)  # (8, 4)
    reconstructed_vec = grad_matrix_full @ full_lambdas

    # 计算总误差 (L2 norm)
    error_norm = np.linalg.norm(reconstructed_vec - target_vec)

    return full_lambdas, reconstructed_vec, error_norm


def reverse_engineer_lambdas(alpha):
    print(f"\n{'=' * 60}")
    print(f" 反向推导 Lambda 组合 (Alpha={alpha}) ")
    print(f"{'=' * 60}")

    # 1. 获取 Correlation 点 P
    p_input = get_tilted_expectations(alpha)
    p_array = np.array(p_input)

    # 2. 计算所有 4 个梯度
    # 固定顺序: 0:(1,1), 1:(1,-1), 2:(-1,1), 3:(-1,-1)
    combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    all_grads = []

    print("正在计算所有 4 个梯度...")
    for i, (s, t) in enumerate(combinations):
        g = get_gradient_at_point(p_array, s, t)
        all_grads.append(g)

    # 3. 遍历所有可能的 3 个方程组合 (C(4,3) = 4 种情况)
    # 索引列表: [0, 1, 2, 3]
    indices = [0, 1, 2, 3]

    best_result = None
    min_error = float('inf')

    print("\n>>> 开始尝试所有方程组合 (Target: Tilted-CHSH) <<<")

    for combo in itertools.combinations(indices, 3):
        combo_name = str(tuple(combinations[i] for i in combo))  # 显示 (s,t)
        print(f"\n--- 尝试组合: 选中索引={combo} (对应方程 (s,t)={combo_name}) ---")

        # 获取完整的 4 元素 Lambda
        full_lambdas, reconstructed, error = solve_lambdas_for_combination(all_grads, combo, alpha)

        # 打印分析
        print(f"  解出的完整 Lambda (4元素): {full_lambdas}")
        print(f"  重构误差 (L2 Norm): {error:.2e}")

        # 检查关键的 0 系数项 (A1, B0, B1) 即索引 1, 2, 3
        zero_terms = reconstructed[1:4]
        print(f"  边缘项残留 (A1, B0, B1): {zero_terms}")
        # 检查 A0 系数
        print(f"  A0 系数 (Target={alpha}): {reconstructed[0]:.6f}")

        # 保存最佳结果
        if error < min_error:
            min_error = error
            best_result = {
                'indices': combo,
                'full_lambdas': full_lambdas,
                'reconstructed': reconstructed
            }

    # 4. 输出最终结论
    print(f"\n{'=' * 60}")
    print(" 最佳推导结果 ")
    print(f"{'=' * 60}")

    if best_result:
        idxs = best_result['indices']

        print(f"最佳方程组合索引: {idxs}")
        print(f"对应被舍弃的索引: {list(set(indices) - set(idxs))}")
        print(f"最佳权重 Lambda (Full 4-element): {best_result['full_lambdas']}")
        print(f"最小误差: {min_error:.2e}")

        vec = best_result['reconstructed']
        print("\n重建的 Bell 不等式系数:")
        print(f"  A0: {vec[0]:.6f} (Alpha)")
        print(f"  A1: {vec[1]:.6f} (Should be 0)")
        print(f"  B0: {vec[2]:.6f} (Should be 0)")
        print(f"  B1: {vec[3]:.6f} (Should be 0)")
        print(f"  E00: {vec[4]:.6f}")
        print(f"  E01: {vec[5]:.6f}")
        print(f"  E10: {vec[6]:.6f}")
        print(f"  E11: {vec[7]:.6f}")

        # 归一化展示 (以 E00 为基准)
        norm_factor = vec[4]
        if abs(norm_factor) > 1e-9:
            norm_vec = vec / norm_factor
            print("\n归一化后的系数 (除以 E00):")
            print(np.round(norm_vec, 6))
        else:
            print("\n无法归一化 (E00 系数过小)")
    else:
        print("未找到有效解。")


if __name__ == "__main__":
    # 设定 alpha
    curr_alpha = 0.5
    reverse_engineer_lambdas(curr_alpha)