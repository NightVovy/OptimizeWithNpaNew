import numpy as np
import cvxpy as cp
import sys

# 导入依赖
try:
    from NPAHierarchy_5 import npa_constraints # 有改动
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
    M_fc (3 rows, 4 cols), B2 列为 0。
    """
    if len(vec) != 8:
        raise ValueError("系数向量必须包含 8 个元素。")

    A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1 = vec

    M = np.zeros((3, 4))
    M[0, 1], M[0, 2] = B0, B1
    M[1, 0] = A0
    M[1, 1], M[1, 2] = A0B0, A0B1
    M[2, 0] = A1
    M[2, 1], M[2, 2] = A1B0, A1B1
    return M


def calculate_classical_bound(coeff_matrix):
    """
    计算经典界 (忽略 B2)。
    """
    max_val = -np.inf
    for a0 in [-1, 1]:
        for a1 in [-1, 1]:
            for b0 in [-1, 1]:
                for b1 in [-1, 1]:
                    val = 0
                    val += coeff_matrix[0, 0]
                    val += coeff_matrix[0, 1] * b0 + coeff_matrix[0, 2] * b1
                    val += coeff_matrix[1, 0] * a0 + coeff_matrix[2, 0] * a1
                    val += coeff_matrix[1, 1] * a0 * b0 + coeff_matrix[1, 2] * a0 * b1
                    val += coeff_matrix[2, 1] * a1 * b0 + coeff_matrix[2, 2] * a1 * b1
                    if val > max_val:
                        max_val = val
    return max_val


# ==============================================================================
# 2. 约束构建器：通用 Bell 不等式
# ==============================================================================
def build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix):
    expr = 0
    if coeff_matrix[0, 0] != 0:
        expr += coeff_matrix[0, 0]

    def get_term_E(alice_list, bob_list):
        t = get_gamma_element(G, ind_catalog, m_vec, alice_list, bob_list)
        if t is None:
            raise ValueError(f"约束构建失败: 无法在 Gamma 矩阵中找到算符 A{alice_list}B{bob_list}。")
        return t

    for y_idx in range(2):
        c = coeff_matrix[0, y_idx + 1]
        if c != 0:
            term_Fy = get_term_E([], [2 + y_idx])
            expr += c * (2 * term_Fy - 1)

    for x_idx in range(2):
        c = coeff_matrix[x_idx + 1, 0]
        if c != 0:
            term_Ex = get_term_E([x_idx], [])
            expr += c * (2 * term_Ex - 1)

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
def build_fidelity_objective(G, ind_catalog, m_vec, theta):
    """
    构建 Tilted CHSH 场景下的 Fidelity 目标函数。
    直接使用 theta 计算系数。
    """

    # 计算系数
    cos2theta = np.cos(2 * theta)
    sin2theta = np.sin(2 * theta)

    # 辅助查找
    def get_term(alice_seq, bob_seq):
        val = get_gamma_element(G, ind_catalog, m_vec, alice_seq, bob_seq)
        if val is None:
            raise ValueError(f"目标函数构建失败: Term <A{alice_seq} B{bob_seq}> not found。请检查 k 值。")
        return val

    # 索引定义
    MA = m_vec[0]
    idx_A0, idx_A1 = 0, 1
    idx_B0, idx_B1, idx_B2 = MA, MA + 1, MA + 2
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

    # ==========================================================================
    # 组装最终 func
    # ==========================================================================

    part_1 = 0.25 * (1 + d_A0B0 + cos2theta * (d_A0 + d_B0))

    part_2 = (sin2theta / 16) * (
            d_A1B2
            - d_A1B0B2B0
            - d_A0A1A0B2
            + d_A0A1A0B0B2B0
    )

    part_3 = - (sin2theta / 16) * (
            d_A0A1B0B2
            - d_A0A1B2B0
            - d_A1A0B0B2
            + d_A1A0B2B0
    )

    # 返回元组以便调试查看各部分，最后再求和
    return part_1, part_2, part_3


# ==============================================================================
# 4. 主计算函数
# ==============================================================================
def calculate_fidelity(target_val: float, coeff_matrix: np.ndarray, desc: np.ndarray, theta: float,
                       k: int = 3):
    m_vec = desc[2:4]

    # 1. 根据 theta 反推 alpha (用于 Localizing Matrix 和 不等式系数构建)
    # sin(2theta) = sqrt((1 - x) / (1 + x)) 其中 x = alpha^2 / 4
    # 推导得: alpha = 2 * cos(2theta) / sqrt(1 + sin^2(2theta))
    sin_2theta = np.sin(2 * theta)
    cos_2theta = np.cos(2 * theta)
    alpha = 2 * cos_2theta / np.sqrt(1 + sin_2theta ** 2)

    # 计算 mu (用于打印信息，localizing matrix 内部也会算)
    mu = np.arctan(sin_2theta)

    print(f"[Calc Params] Theta={theta:.4f} -> Alpha={alpha:.4f}, Mu={mu:.4f}")

    # 2. 构建 NPA 变量
    dummy_cg = np.zeros((3, 4))
    G, constraints, ind_catalog = npa_constraints(dummy_cg, desc, k=k, enforce_data=False)

    # 3. [Constraint 1] Bell 不等式约束
    bell_expr = build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix)
    constraints.append(bell_expr == target_val)

    # 4. [Constraint 2] Localizing Matrix 约束
    try:
        # 传入计算出的 alpha
        loc_constraints = build_localizing_matrix_constraint(G, ind_catalog, m_vec, alpha)
        constraints.extend(loc_constraints)
    except Exception as e:
        print(f"Localizing Matrix 构建失败: {e}")
        return None

    # 5. 构造 Fidelity 目标 (传入 theta)
    try:
        part_1, part_2, part_3 = build_fidelity_objective(G, ind_catalog, m_vec, theta)
        func = part_1 + part_2 + part_3
    except ValueError as e:
        print(f"目标函数构建失败: {e}")
        return None

    # 6. 求解
    print(f"Solving SDP for Target={target_val:.4f}...")
    prob = cp.Problem(cp.Minimize(func), constraints)

    try:
        # prob.solve(solver=cp.SCS, eps=1e-7, max_iters=100000)
        prob.solve(solver=cp.MOSEK, verbose=True)
    except cp.SolverError:
        return None

    # [调试打印]
    print(f"Solver Status: {prob.status}")  # <--- 看看这里输出了什么？
    # ==============================================================================
    # [调试点] 在此处设置断点 (Line ~230)
    # ==============================================================================
    # 1. 查看 func 的各部分值:
    #    在 Debugger 的 Evaluate Expression 中输入: part_1.value, part_2.value, part_3.value
    # 2. 查看 Gamma 矩阵:
    #    在 Debugger 中找到 G，查看 G.value (View as Array)
    # 3. 查看 Localizing Matrix:
    #    Loc 矩阵是约束的一部分。
    #    在 Debugger 中找到 loc_constraints 列表 (通常是 constraints 列表的最后几个)。
    #    输入: loc_constraints[0].args[0].value  (查看约束矩阵的数值)
    #         或者 constraints[-1].args[0].value

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return prob.value
    else:
        return None


# ==============================================================================
# 5. 主程序入口
# ==============================================================================
if __name__ == '__main__':
    # 场景: Alice 2, Bob 3
    desc = np.array([2, 2, 2, 3])
    # k_level = 3
    k_level = "1+aaa+bbb"

    # 1. 输入 Theta
    theta = np.pi / 6

    # 2. 根据 Theta 计算对应的 Alpha (为了构建 inequality 系数)
    # 这里的计算和函数内部一样，仅仅是为了生成 vec_ineq2
    sin_2theta = np.sin(2 * theta)
    cos_2theta = np.cos(2 * theta)
    alpha_val = 2 * cos_2theta / np.sqrt(1 + sin_2theta ** 2)

    # 3. 输入系数矩阵 (Tilted CHSH)
    # 系数: [alpha, 0, 0, 0, 1, 1, 1, -1]
    vec_ineq2 = [alpha_val, 0, 0, 0, 1., 1., 1., -1.]
    coeff_ineq2 = convert_vector_to_matrix(vec_ineq2)

    # 4. 计算边界
    # Tilted CHSH 量子最大违背: sqrt(8 + 2*alpha^2)
    quantum_bound = np.sqrt(8 + 2 * alpha_val ** 2)

    print(f"\n[Main] Theta={theta:.4f}, Derived Alpha={alpha_val:.4f}")
    print(f"Quantum Bound = {quantum_bound:.6f}")

    # 5. 测试
    fid_max = calculate_fidelity(quantum_bound, coeff_ineq2, desc, theta, k=k_level)

    if fid_max is not None:
        print(f"Min Fidelity: {fid_max:.8f}")
    else:
        print("Calculation Failed.")