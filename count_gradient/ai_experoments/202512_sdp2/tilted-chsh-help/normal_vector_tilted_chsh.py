import numpy as np
from math import asin, pi

# 尝试导入 tilted-alpha-correlation 中的函数
# 确保 tilted_alpha_correlation.py 在同一目录下
try:
    from tilted_alpha_correlation import get_tilted_expectations
except ImportError:
    # 为了防止作为模块被导入时报错（如果调用方不需要此函数，可以忽略）
    pass


# ==========================================
# 第一部分：数学定义 (复用参考代码)
# ==========================================

def compute_arcsin_term(Ax, alpha, By, s_or_t, Exy):
    """
    辅助函数：计算单个 arcsin 项
    公式: arcsin[(Exy + s*By) / (1 + s*Ax)]
    """
    numerator = Exy + s_or_t * By
    denominator = 1 + s_or_t * Ax
    val = numerator / denominator
    val = np.clip(val, -1.0, 1.0)
    return asin(val)


def constraint_equation(p, s, t):
    """
    约束方程定义。
    公式: term1 + term2 + term3 - term4 = π
    """
    A0, A1, B0, B1, E00, E01, E10, E11 = p
    term1 = compute_arcsin_term(A0, s, B0, s, E00)
    term2 = compute_arcsin_term(A1, t, B0, t, E10)
    term3 = compute_arcsin_term(A0, s, B1, s, E01)
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
# 第二部分：核心逻辑处理 (接口已更新为适配 4 Lambdas)
# ==========================================

def get_normalized_bell_coeffs(alpha, lambdas):
    """
    计算给定 alpha 和 lambda 组合下的归一化 Bell 不等式系数。

    Args:
        alpha (float): Tilted-CHSH 参数
        lambdas (list): 4 个权重值的列表 [l1, l2, l3, l4]
                        对应组合: (1,1), (1,-1), (-1,1), (-1,-1)

    Returns:
        np.array: 归一化后的 8 个系数 [A0, A1, B0, B1, E00, E01, E10, E11]
    """
    # 1. 验证 Lambda 长度及零值约束
    # 修正点：这里必须检查长度为 4，而不是 3
    if len(lambdas) != 4:
        raise ValueError(f"Lambda 长度必须为 4，当前长度为 {len(lambdas)}。")

    # [新增检查] 验证是否至少有一个 lambda 接近 0
    # 我们认为 < 1e-6 即为 "差距极小" 或 "严格为0"
    is_valid_basis = any(abs(l) < 1e-6 for l in lambdas)
    if not is_valid_basis:
        raise ValueError("输入的 4 个 Lambda 中必须至少有一个为 0 (或极接近 0)，以保证该不等式是由 3 个独立方程构成的。")

    # 2. 获取 Correlation 点
    try:
        from tilted_alpha_correlation import get_tilted_expectations
    except ImportError:
        raise ImportError("无法导入 get_tilted_expectations，请确保文件存在。")

    p_input = get_tilted_expectations(alpha)

    # 3. 计算所有 4 个方程的梯度
    # 组合顺序对应: l1->(1,1), l2->(1,-1), l3->(-1,1), l4->(-1,-1)
    combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    grads = []

    for s, t in combinations:
        g = get_gradient_at_point(p_input, s, t)
        grads.append(g)

    # 4. 加权求和 (遍历 4 个梯度)
    n_vector = np.zeros(8)
    for i in range(4):
        # 即使 lambda 为 0 也进行计算，保持逻辑一致性
        n_vector += lambdas[i] * grads[i]

    # 5. 归一化 (以绝对值最大的元素为基准)
    max_val = np.max(np.abs(n_vector))
    if max_val < 1e-9:
        print("警告: 梯度组合结果接近零向量")
        return n_vector

    normalized_n = n_vector / max_val
    return normalized_n


def analyze_and_compute_normal(alpha, lambdas):
    """
    原有的打印展示函数 (保留用于独立运行)
    """
    print(f"当前 alpha 值: {alpha}")

    try:
        norm_coeffs = get_normalized_bell_coeffs(alpha, lambdas)

        print(f"\n[构造 Bell 不等式]")
        print(f"  使用权重 lambda: {lambdas}")
        print("\n  归一化后的系数 (Normalized by max value):")
        print("  ", np.round(norm_coeffs, 6))

        print(f"\n  [观察] A0 系数: {norm_coeffs[0]:.6f}")

    except Exception as e:
        print(f"发生错误: {e}")


# ==========================================
# 主执行入口
# ==========================================

if __name__ == "__main__":
    # 设置参数
    alpha_val = 0.5
    # 设置权重 (符合 4 个值且含 0 的要求)
    lambda_vals = [0.33, 0.33, 0.34, 0]

    analyze_and_compute_normal(alpha=alpha_val, lambdas=lambda_vals)