import numpy as np
from math import asin, pi
from scipy.optimize import minimize
import sys

# ==============================================================================
# 导入依赖 (确保相关文件在同一目录下)
# ==============================================================================
try:
    from tilted_alpha_correlation import get_tilted_expectations
    from BellInequalityMax_new import compute_bell_limits
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 'tilted_alpha_correlation.py' 和 'BellInequalityMax_new.py' 都在当前目录下。")
    sys.exit(1)


# ==============================================================================
# 第一部分：法向量(梯度)计算逻辑
# (复用 normal_vector_tilted_chsh.py 的核心逻辑以确保一致性)
# ==============================================================================

def compute_arcsin_term(Ax, alpha, By, s_or_t, Exy):
    """
    计算单个 arcsin 项
    公式: arcsin[(Exy + s*By) / (1 + s*Ax)]
    """
    numerator = Exy + s_or_t * By
    denominator = 1 + s_or_t * Ax
    if abs(denominator) < 1e-9:
        val = 0
    else:
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


def get_all_gradients(alpha):
    """
    计算 alpha 下所有 4 种 (s, t) 组合的梯度
    返回: 4x8 的矩阵，每一行对应一个 lambda 分量的法向量
    """
    p_input = get_tilted_expectations(alpha)

    # 组合顺序: (1,1), (1,-1), (-1,1), (-1,-1)
    combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    grads = []

    for s, t in combinations:
        g = get_gradient_at_point(p_input, s, t)
        grads.append(g)

    return np.array(grads)


# ==============================================================================
# 第二部分：优化目标函数
# ==============================================================================

def vector_to_fc_matrix(coeffs):
    """
    将 8 维系数向量转换为 BellInequalityMax_new 需要的 FC 格式矩阵 (3x3)。
    """
    M = np.zeros((3, 3))
    M[1, 0] = coeffs[0]  # A0
    M[2, 0] = coeffs[1]  # A1
    M[0, 1] = coeffs[2]  # B0
    M[0, 2] = coeffs[3]  # B1
    M[1, 1] = coeffs[4]  # E00
    M[1, 2] = coeffs[5]  # E01
    M[2, 1] = coeffs[6]  # E10
    M[2, 2] = coeffs[7]  # E11
    return M


def gap_objective(lambdas, gradients, desc, k_level):
    """
    目标函数：计算 -(S_Q - S_L)。
    """
    # 1. 线性组合法向量得到 Bell 不等式系数
    coeffs = np.dot(lambdas, gradients)

    # 2. 归一化系数
    max_val = np.max(np.abs(coeffs))
    if max_val > 1e-9:
        coeffs = coeffs / max_val

    # 3. 构造输入矩阵
    M_fc = vector_to_fc_matrix(coeffs)

    # 4. 调用计算模块
    try:
        # 这里的 solver 如果没有安装 SCS，可以尝试 'CVXOPT' 或 'ECOS'，或者留空让 cvxpy 自动选择
        b_cl, b_qm = compute_bell_limits(M_fc, desc, notation='fc', k=k_level, verbose=False, solver='SCS')

        if b_cl is None or b_qm is None:
            return 1e5

        gap = b_qm - b_cl
        return -gap  # 最小化负gap = 最大化gap
    except Exception:
        return 1e5


# ==============================================================================
# 第三部分：主执行逻辑
# ==============================================================================

def optimize_lambda(alpha=0.5):
    print(f"=== 开始寻找最优 Lambda 组合 (Alpha={alpha}) ===")

    # 1. 准备梯度矩阵 (4x8)
    print("正在计算基础梯度 (Normal Vectors)...")
    gradients = get_all_gradients(alpha)

    # 2. 生成随机初始猜测 (Random Initialization)
    # --------------------------------------------------------------------------
    # 使用 numpy 生成 4 个随机数，并归一化，保证 sum=1 且 > 0
    rng = np.random.default_rng()
    raw_x0 = rng.random(4)
    x0 = raw_x0 / np.sum(raw_x0)

    print(f"生成的随机初始猜测 x0: {np.round(x0, 6)}")
    # --------------------------------------------------------------------------

    # 约束: sum(lambda) = 1
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # 边界: lambda_i >= 0
    bnds = tuple([(0, None) for _ in range(4)])

    desc = [2, 2, 2, 2]
    k_level = 1  # 如需更高精度，请改为 2 或 3

    print("开始 SLSQP 优化 (目标: Maximize Gap = S_Q - S_L)...")

    # 3. 执行优化
    res = minimize(
        gap_objective,
        x0,
        args=(gradients, desc, k_level),
        method='SLSQP',
        bounds=bnds,
        constraints=cons,
        options={'ftol': 1e-6, 'disp': True}
    )

    if not res.success:
        print("警告: 优化未收敛!")
        print(f"原因: {res.message}")
    else:
        print("优化成功收敛。")

    # ==========================================================================
    # 结果分析与输出 (包含稀疏性处理)
    # ==========================================================================
    best_lambda = res.x

    # 稀疏性处理: 将极小值置为 0 (阈值设为 1e-5)
    clean_lambda = np.array([0.0 if x < 1e-5 else x for x in best_lambda])

    # 重新归一化以便展示 (保持严谨性)
    if np.sum(clean_lambda) > 0:
        clean_lambda = clean_lambda / np.sum(clean_lambda)

    print("\n" + "=" * 40)
    print("最终结果分析")
    print("=" * 40)

    print(f"原始最优 Lambda: {np.round(best_lambda, 6)}")
    print(f"稀疏化 Lambda  : {np.round(clean_lambda, 6)}")

    # 验证哪个分量为 0 (线性相关性)
    zero_indices = np.where(clean_lambda == 0)[0]
    combo_names = ["(1,1)", "(1,-1)", "(-1,1)", "(-1,-1)"]
    if len(zero_indices) > 0:
        print(f"被排除的法向量索引 (Lambda ≈ 0): {zero_indices}")
        print(f"对应 (s,t) 组合: {[combo_names[i] for i in zero_indices]}")
    else:
        print("所有法向量均有贡献 (未发现明显的线性依赖排除)。")

    # 计算最终的 Gap 和各项数值
    final_coeffs = np.dot(clean_lambda, gradients)
    # 归一化系数方便阅读
    max_coeff = np.max(np.abs(final_coeffs))
    if max_coeff > 1e-9:
        final_coeffs = final_coeffs / max_coeff

    M_final = vector_to_fc_matrix(final_coeffs)
    s_cl, s_qm = compute_bell_limits(M_final, desc, notation='fc', k=k_level, verbose=False, solver='SCS')

    print("-" * 30)
    print(f"构建的 Bell 不等式系数 (归一化):\n{np.round(final_coeffs, 5)}")
    print(f"顺序: [A0, A1, B0, B1, E00, E01, E10, E11]")
    print("-" * 30)
    print(f"经典界 (S_L) : {s_cl:.6f}")
    print(f"量子界 (S_Q) : {s_qm:.6f}")
    print(f"最大 Gap     : {(s_qm - s_cl):.6f}")
    print("=" * 40)


if __name__ == "__main__":
    optimize_lambda(alpha=0.5)