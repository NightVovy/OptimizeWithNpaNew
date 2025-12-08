import numpy as np
import cvxpy as cp
import sys

# 导入依赖
try:
    from NPAHierarchy_3 import npa_constraints
    from locate_gamma_2 import get_gamma_element
    from localizing_matrix import build_localizing_matrix_constraint
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


# ==============================================================================
# 1. 工具函数：系数转换与经典界计算
# ==============================================================================
def convert_vector_to_matrix(vec):
    """
    将 8 元素向量转换为 3x4 Full Correlation 系数矩阵。
    Vector: [A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1]
    M_fc (3 rows, 4 cols):
        [K,  B0,   B1,   B2]
        [A0, A0B0, A0B1, A0B2]
        [A1, A1B0, A1B1, A1B2]

    注意：涉及 B2 的系数默认为 0。
    """
    if len(vec) != 8:
        raise ValueError("系数向量必须包含 8 个元素。")

    A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1 = vec

    # 初始化 3x4 矩阵
    M = np.zeros((3, 4))

    # 填入标准 CHSH 相关项
    M[0, 1], M[0, 2] = B0, B1
    M[1, 0] = A0
    M[1, 1], M[1, 2] = A0B0, A0B1
    M[2, 0] = A1
    M[2, 1], M[2, 2] = A1B0, A1B1

    # 第 4 列 (B2 相关) 保持为 0
    return M


def calculate_classical_bound(coeff_matrix):
    """
    遍历所有确定性策略，计算给定不等式的经典界 (Local Bound)。
    注意：在计算经典界时，不考虑 B2 (Col 3)，只遍历 A0, A1, B0, B1。
    """
    max_val = -np.inf
    # 遍历 A0, A1, B0, B1 的所有 16 种组合
    for a0 in [-1, 1]:
        for a1 in [-1, 1]:
            for b0 in [-1, 1]:
                for b1 in [-1, 1]:
                    # 计算当前策略下的不等式值 (忽略 B2)
                    val = 0
                    val += coeff_matrix[0, 0]  # K

                    # 单体项
                    val += coeff_matrix[0, 1] * b0 + coeff_matrix[0, 2] * b1
                    val += coeff_matrix[1, 0] * a0 + coeff_matrix[2, 0] * a1

                    # 关联项
                    val += coeff_matrix[1, 1] * a0 * b0 + coeff_matrix[1, 2] * a0 * b1
                    val += coeff_matrix[2, 1] * a1 * b0 + coeff_matrix[2, 2] * a1 * b1

                    if val > max_val:
                        max_val = val
    return max_val


# ==============================================================================
# 2. 约束构建器：通用 Bell 不等式 (保持不变，无需考虑 B2)
# ==============================================================================
def build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix):
    """
    根据输入的系数矩阵，构建用于 SDP 约束的 Bell 表达式。
    coeff_matrix 可以是 3x3 或 3x4，如果 B2 系数为0，逻辑兼容。
    """
    expr = 0
    if coeff_matrix[0, 0] != 0:
        expr += coeff_matrix[0, 0]

    def get_term_E(alice_list, bob_list):
        t = get_gamma_element(G, ind_catalog, m_vec, alice_list, bob_list)
        if t is None:
            raise ValueError(f"约束构建失败: 无法在 Gamma 矩阵中找到算符 A{alice_list}B{bob_list}。")
        return t

    # Bob 单体项 (只遍历 B0, B1，即 Col 1, 2)
    # 假设输入的不等式不涉及 B2
    for y_idx in range(2):
        c = coeff_matrix[0, y_idx + 1]
        if c != 0:
            term_Fy = get_term_E([], [2 + y_idx])
            expr += c * (2 * term_Fy - 1)

    # Alice 单体项
    for x_idx in range(2):
        c = coeff_matrix[x_idx + 1, 0]
        if c != 0:
            term_Ex = get_term_E([x_idx], [])
            expr += c * (2 * term_Ex - 1)

    # 关联项
    for x_idx in range(2):
        for y_idx in range(2):
            c = coeff_matrix[x_idx + 1, y_idx + 1]
            if c != 0:
                bob_setting = [2 + y_idx]
                term_ExFy = get_term_E([x_idx], bob_setting)
                term_Ex = get_term_E([x_idx], [])
                term_Fy = get_term_E([], bob_setting)
                term_AxBy = 4 * term_ExFy - 2 * term_Ex - 2 * term_Fy + 1
                expr += c * term_AxBy
    return expr


# ==============================================================================
# 3. 目标函数构建器：Tilted CHSH Fidelity
# ==============================================================================
def build_fidelity_objective(G, ind_catalog, m_vec, alpha):
    """
    构建 Tilted CHSH 场景下的 Fidelity 目标函数。
    包含 A0, A1 以及 B0, B1, B2。
    """

    # --- (A) 计算 Theta ---
    # sin(2theta) = sqrt((1 - alpha^2/4) / (1 + alpha^2/4))
    val_sin_2theta = np.sqrt((1 - (alpha ** 2) / 4) / (1 + (alpha ** 2) / 4))

    # theta = 0.5 * arcsin(...)
    theta = 0.5 * np.arcsin(val_sin_2theta)

    cos2theta = np.cos(2 * theta)
    sin2theta = np.sin(2 * theta)

    print(f"[Info] Alpha={alpha}, 2Theta={np.degrees(2*theta):.2f} deg")

    # --- (B) 辅助查找 ---
    def get_term(alice_seq, bob_seq):
        val = get_gamma_element(G, ind_catalog, m_vec, alice_seq, bob_seq)
        if val is None:
            raise ValueError(f"目标函数构建失败: Term <A{alice_seq} B{bob_seq}> not found.")
        return val

    # --- (C) 索引定义 (基于 m_vec) ---
    MA = m_vec[0]  # Alice 设置数 (通常为2)

    # Alice 索引
    idx_A0 = 0
    idx_A1 = 1

    # Bob 索引 (从 MA 开始)
    idx_B0 = MA
    idx_B1 = MA + 1
    idx_B2 = MA + 2  # B2

    idx_I = []

    # ==========================================================================
    # --- (D) 投影算子组合 (Terms) ---
    # ==========================================================================

    # --- 3x3 ---
    term_E0E1E0_F0F2F0 = get_term([idx_A0, idx_A1, idx_A0], [idx_B0, idx_B2, idx_B0])

    # --- 3x2 ---
    term_E0E1E0_F0F2 = get_term([idx_A0, idx_A1, idx_A0], [idx_B0, idx_B2])
    term_E0E1E0_F2F0 = get_term([idx_A0, idx_A1, idx_A0], [idx_B2, idx_B0])

    term_E0E1_F0F2F0 = get_term([idx_A0, idx_A1], [idx_B0, idx_B2, idx_B0])
    term_E1E0_F0F2F0 = get_term([idx_A1, idx_A0], [idx_B0, idx_B2, idx_B0])

    # --- 1+3 / 2+2 / 3+1 ---
    term_E0E1E0_F2 = get_term([idx_A0, idx_A1, idx_A0], [idx_B2])
    term_E1_F0F2F0 = get_term([idx_A1], [idx_B0, idx_B2, idx_B0])

    term_E0E1_F0F2 = get_term([idx_A0, idx_A1], [idx_B0, idx_B2])
    term_E0E1_F2F0 = get_term([idx_A0, idx_A1], [idx_B2, idx_B0])
    term_E1E0_F0F2 = get_term([idx_A1, idx_A0], [idx_B0, idx_B2])
    term_E1E0_F2F0 = get_term([idx_A1, idx_A0], [idx_B2, idx_B0])

    # --- 2+1 / 1+2 ---
    term_E0E1_F0 = get_term([idx_A0, idx_A1], [idx_B0])
    term_E0E1_F2 = get_term([idx_A0, idx_A1], [idx_B2])
    term_E1E0_F0 = get_term([idx_A1, idx_A0], [idx_B0])
    term_E1E0_F2 = get_term([idx_A1, idx_A0], [idx_B2])

    term_E0_F0F2 = get_term([idx_A0], [idx_B0, idx_B2])
    term_E0_F2F0 = get_term([idx_A0], [idx_B2, idx_B0])
    term_E1_F0F2 = get_term([idx_A1], [idx_B0, idx_B2])
    term_E1_F2F0 = get_term([idx_A1], [idx_B2, idx_B0])

    # --- 3 (Pure Terms) ---
    term_E0E1E0 = get_term([idx_A0, idx_A1, idx_A0], idx_I)
    term_F0F2F0 = get_term(idx_I, [idx_B0, idx_B2, idx_B0])

    # --- 2 / 1+1 ---
    term_E1_F2 = get_term([idx_A1], [idx_B2])
    term_E0_F0 = get_term([idx_A0], [idx_B0])
    term_E0_F2 = get_term([idx_A0], [idx_B2])
    term_E1_F0 = get_term([idx_A1], [idx_B0])

    term_E0E1 = get_term([idx_A0, idx_A1], idx_I)
    term_E1E0 = get_term([idx_A1, idx_A0], idx_I)

    term_F0F2 = get_term(idx_I, [idx_B0, idx_B2])
    term_F2F0 = get_term(idx_I, [idx_B2, idx_B0])

    # --- 1 (Pure Singles) ---
    term_E0 = get_term([idx_A0], idx_I)
    term_E1 = get_term([idx_A1], idx_I)
    term_F0 = get_term(idx_I, [idx_B0])
    term_F2 = get_term(idx_I, [idx_B2])

    term_I = get_term(idx_I, idx_I)


    # --- (E) 可观测量组合  ---
    # d_X = LinearComb(terms)

    # d_A0 = ...
    d_A0 = 2 * term_E0 - term_I
    d_B0 = 2 * term_F0 - term_I

    d_A0B0 = 4 * term_E0_F0 - 2 * term_E0 - 2 * term_F0 + term_I
    d_A1B2 = 4 * term_E1_F2 - 2 * term_E1 - 2 * term_F2 + term_I

    d_A1B0B2B0 = 16 * term_E1_F0F2F0 - 8 * (term_E1_F0F2 + term_E1_F2F0 + term_F0F2F0)
    + 4 * (term_E1_F2 + term_F0F2 + term_F2F0)
    - 2 * (term_E1 + term_F2) + term_I

    d_A0A1A0B2 = 16 * term_E0E1E0_F2 - 8 * (term_E0E1_F2 + term_E1E0_F2 + term_E0E1E0)
    + 4 * (term_E1_F2 + term_E0E1 + term_E1E0)
    - 2 * (term_E1 + term_F2) + term_I


    d_A0A1B0B2 = 16 * term_E0E1_F0F2
    - 8 * (term_E0E1_F0 + term_E0E1_F2 + term_E0_F0F2 + term_E1_F0F2)
    + 4 * (term_E0E1 + term_F0F2 + term_E0_F0 + term_E0_F2 + term_E1_F0 + term_E1_F2)
    - 2 * (term_E0 + term_E1 + term_F0 + term_F2) + term_I

    d_A0A1B2B0 = 16 * term_E0E1_F2F0
    - 8 * (term_E0E1_F2 + term_E0E1_F0 + term_E0_F2F0 + term_E1_F2F0)
    + 4 * (term_E0E1 + term_F2F0 + term_E0_F2 + term_E0_F0 + term_E1_F2 + term_E1_F0)
    - 2 * (term_E0 + term_E1 + term_F2 + term_F0) + term_I

    d_A1A0B0B2 = 16 * term_E1E0_F0F2
    - 8 * (term_E1E0_F0 + term_E1E0_F2 + term_E1_F0F2 + term_E0_F0F2)
    + 4 * (term_E1E0 + term_F0F2 + term_E1_F0 + term_E1_F2 + term_E0_F0 + term_E0_F2)
    - 2 * (term_E1 + term_E0 + term_F0 + term_F2) + term_I

    d_A1A0B2B0 = 16 * term_E1E0_F2F0
    - 8 * (term_E1E0_F2 + term_E1E0_F0 + term_E1_F2F0 + term_E0_F2F0)
    + 4 * (term_E1E0 + term_F2F0 + term_E1_F2 + term_E1_F0 + term_E0_F2 + term_E0_F0)
    - 2 * (term_E1 + term_E0 + term_F2 + term_F0) + term_I

    d_A0A1A0B0B2B0 = 64 * (term_E0E1E0_F0F2F0)
    - 32 * (term_E0E1E0_F0F2 + term_E0E1E0_F2F0 + term_E0E1_F0F2F0 + term_E1E0_F0F2F0)
    + 16 * (term_E0E1E0_F2 + term_E1_F0F2F0 + term_E0E1_F0F2 + term_E0E1_F2F0 + term_E1E0_F0F2 + term_E1E0_F2F0)
    - 8 * (term_E0E1E0 + term_F0F2F0 + term_E0E1_F2 + term_E1E0_F2 + term_E1_F0F2 + term_E1_F2F0)
    + 4 * (term_E0E1 + term_E1E0 + term_F0F2 + term_F2F0 + term_E1_F2)
    - 2 * (term_E1 + term_F2) + term_I

    # --- (F) 组装目标函数 ---
    # func = 1/2 * (1 + d_A0B0 + cos2theta * (d_A0 + d_B0))
    #      + sin2theta/8 * (d_A1B2 - d_A1B0B2B0 - d_A0A1A0B2 + d_A0A1A0B0B2B0)
    #      - sin2theta/8 * (d_A0A1B0B2 - d_A0A1B2B0 - d_A1A0B0B2 + d_A1A0B2B0)

    part_1 = 0.5 * (1 + d_A0B0 + cos2theta * (d_A0 + d_B0))

    part_2 = (sin2theta / 8) * (
            d_A1B2
            - d_A1B0B2B0
            - d_A0A1A0B2
            + d_A0A1A0B0B2B0
    )

    part_3 = - (sin2theta / 8) * (
            d_A0A1B0B2
            - d_A0A1B2B0
            - d_A1A0B0B2
            + d_A1A0B2B0
    )

    func = part_1 + part_2 + part_3
    return func


# ==============================================================================
# 4. 主计算函数
# ==============================================================================
def calculate_fidelity(target_val: float, coeff_matrix: np.ndarray, desc: np.ndarray, alpha: float,
                       k: int = 3) -> float:
    m_vec = desc[2:4]

    # 1. 哑数据构建变量
    # dummy_cg 需要是 3x4 的 (因为 Bob 有 3 个输入)
    dummy_cg = np.zeros((3, 4))
    G, constraints, ind_catalog = npa_constraints(dummy_cg, desc, k=k, enforce_data=False)

    # 2. [Constraint 1] Bell 不等式约束
    bell_expr = build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix)
    constraints.append(bell_expr == target_val)

    # 3. [Constraint 2] Localizing Matrix 约束
    # 调用 localizing_matrix.py
    try:
        loc_constraints = build_localizing_matrix_constraint(G, ind_catalog, m_vec, alpha)
        constraints.extend(loc_constraints)
    except Exception as e:
        print(f"Localizing Matrix 构建失败: {e}")
        return None

    # 4. 构造 Fidelity 目标
    try:
        func = build_fidelity_objective(G, ind_catalog, m_vec, alpha)
    except ValueError as e:
        print(f"目标函数构建失败 (K可能不足): {e}")
        return None

    # 5. 求解
    print(f"Solving SDP for Target={target_val:.4f}, Alpha={alpha}...")
    prob = cp.Problem(cp.Minimize(func), constraints)

    try:
        prob.solve(solver=cp.SCS, eps=1e-6, max_iters=20000)
    except cp.SolverError:
        return None

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return prob.value
    else:
        return None


# ==============================================================================
# 5. 主程序入口
# ==============================================================================
if __name__ == '__main__':
    # 场景配置: Alice 2 inputs, Bob 3 inputs (B0, B1, B2)
    desc = np.array([2, 2, 2, 3])
    k_level = 3

    # 设置 Alpha
    alpha = 0.5

    # --- 输入系数矩阵 (Inequality 2) ---
    # Vector: [A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1]
    vec_ineq2 = [alpha, 0, 0, 0, 1., 1., 1., -1.]
    coeff_ineq2 = convert_vector_to_matrix(vec_ineq2)

    # --- 计算边界 ---
    # 不等式2的最大量子违背
    quantum_bound = np.sqrt(8 + 2 * alpha**2)
    local_bound = calculate_classical_bound(coeff_ineq2)

    print(f"\n[Analysis] Alpha = {alpha}")
    print(f"Classical Bound = {local_bound:.6f}")
    print(f"Quantum Bound   = {quantum_bound:.6f}")

    # --- 测试 ---
    print(f"\n=== Test Case: Target Violation = Quantum Bound ===")
    fid_max = calculate_fidelity(quantum_bound, coeff_ineq2, desc, alpha, k=k_level)

    if fid_max is not None:
        print(f"Min Fidelity: {fid_max:.8f}")
    else:
        print("Calculation Failed.")