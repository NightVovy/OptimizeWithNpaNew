import numpy as np
import cvxpy as cp
import sys

# 导入依赖
try:
    # [修改点 1] 导入支持自定义基底的 NPA 版本
    from NPAHierarchy_Custom import npa_constraints
    from locate_gamma_2 import get_gamma_element
    from localizing_matrix import build_localizing_matrix_constraint
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 NPAHierarchy_Custom.py 已保存并在当前目录下。")
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
    """
    cos2theta = np.cos(2 * theta)
    sin2theta = np.sin(2 * theta)

    def get_term(alice_seq, bob_seq):
        val = get_gamma_element(G, ind_catalog, m_vec, alice_seq, bob_seq)
        if val is None:
            raise ValueError(f"目标函数构建失败: Term <A{alice_seq} B{bob_seq}> not found。请检查基底是否完备。")
        return val

    # 索引定义
    MA = m_vec[0]
    idx_A0, idx_A1 = 0, 1
    idx_B0, idx_B1, idx_B2 = MA, MA + 1, MA + 2
    idx_I = []

    # ==========================================================================
    # (D) 投影算子组合 (Terms)
    # ==========================================================================
    # 注意：这里的组合必须包含在 custom_basis 生成的 Gamma 矩阵中

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

    # --- (E) 可观测量组合 ---
    d_A0 = 2 * term_E0 - term_I
    d_B0 = 2 * term_F0 - term_I

    d_A0B0 = 4 * term_E0_F0 - 2 * term_E0 - 2 * term_F0 + term_I
    d_A1B2 = 4 * term_E1_F2 - 2 * term_E1 - 2 * term_F2 + term_I

    d_A1B0B2B0 = 16 * term_E1_F0F2F0 - 8 * (term_E1_F0F2 + term_E1_F2F0 + term_F0F2F0) \
                 + 4 * (term_E1_F2 + term_F0F2 + term_F2F0) \
                 - 2 * (term_E1 + term_F2) + term_I

    d_A0A1A0B2 = 16 * term_E0E1E0_F2 - 8 * (term_E0E1_F2 + term_E1E0_F2 + term_E0E1E0) \
                 + 4 * (term_E1_F2 + term_E0E1 + term_E1E0) \
                 - 2 * (term_E1 + term_F2) + term_I

    d_A0A1B0B2 = 16 * term_E0E1_F0F2 \
                 - 8 * (term_E0E1_F0 + term_E0E1_F2 + term_E0_F0F2 + term_E1_F0F2) \
                 + 4 * (term_E0E1 + term_F0F2 + term_E0_F0 + term_E0_F2 + term_E1_F0 + term_E1_F2) \
                 - 2 * (term_E0 + term_E1 + term_F0 + term_F2) + term_I

    d_A0A1B2B0 = 16 * term_E0E1_F2F0 \
                 - 8 * (term_E0E1_F2 + term_E0E1_F0 + term_E0_F2F0 + term_E1_F2F0) \
                 + 4 * (term_E0E1 + term_F2F0 + term_E0_F2 + term_E0_F0 + term_E1_F2 + term_E1_F0) \
                 - 2 * (term_E0 + term_E1 + term_F2 + term_F0) + term_I

    d_A1A0B0B2 = 16 * term_E1E0_F0F2 \
                 - 8 * (term_E1E0_F0 + term_E1E0_F2 + term_E1_F0F2 + term_E0_F0F2) \
                 + 4 * (term_E1E0 + term_F0F2 + term_E1_F0 + term_E1_F2 + term_E0_F0 + term_E0_F2) \
                 - 2 * (term_E1 + term_E0 + term_F0 + term_F2) + term_I

    d_A1A0B2B0 = 16 * term_E1E0_F2F0 \
                 - 8 * (term_E1E0_F2 + term_E1E0_F0 + term_E1_F2F0 + term_E0_F2F0) \
                 + 4 * (term_E1E0 + term_F2F0 + term_E1_F2 + term_E1_F0 + term_E0_F2 + term_E0_F0) \
                 - 2 * (term_E1 + term_E0 + term_F2 + term_F0) + term_I

    d_A0A1A0B0B2B0 = 64 * (term_E0E1E0_F0F2F0) \
                     - 32 * (term_E0E1E0_F0F2 + term_E0E1E0_F2F0 + term_E0E1_F0F2F0 + term_E1E0_F0F2F0) \
                     + 16 * (
                                 term_E0E1E0_F2 + term_E1_F0F2F0 + term_E0E1_F0F2 + term_E0E1_F2F0 + term_E1E0_F0F2 + term_E1E0_F2F0) \
                     - 8 * (term_E0E1E0 + term_F0F2F0 + term_E0E1_F2 + term_E1E0_F2 + term_E1_F0F2 + term_E1_F2F0) \
                     + 4 * (term_E0E1 + term_E1E0 + term_F0F2 + term_F2F0 + term_E1_F2) \
                     - 2 * (term_E1 + term_F2) + term_I

    # ==========================================================================
    # 组装最终 func
    # ==========================================================================

    part_1 = 0.25 * (1 + d_A0B0 + cos2theta * (d_A0 + d_B0))  # 这里是 0.5 (1/2) 还是 0.25? 原公式是 1/2

    part_2 = (sin2theta / 16) * (
            d_A1B2
            - d_A1B0B2B0
            - d_A0A1A0B2
            + d_A0A1A0B0B2B0
    )

    part_3 = (sin2theta / 16) * (  # 应有个负号 有改动
            d_A0A1B0B2
            - d_A0A1B2B0
            - d_A1A0B0B2
            + d_A1A0B2B0
    )

    func = part_1 + part_2 + part_3
    return part_1, part_2, part_3


# ==============================================================================
# 4. 主计算函数
# ==============================================================================
def calculate_fidelity(target_val: float, coeff_matrix: np.ndarray, desc: np.ndarray, theta: float):
    m_vec = desc[2:4]

    # 1. 根据 theta 反推 alpha
    sin_2theta = np.sin(2 * theta)
    cos_2theta = np.cos(2 * theta)
    alpha = 2 * cos_2theta / np.sqrt(1 + sin_2theta ** 2)
    mu = np.arctan(sin_2theta)

    print(f"[Calc Params] Theta={theta:.4f} -> Alpha={alpha:.4f}, Mu={mu:.4f}")

    # ==========================================================================
    # [修改点 2] 定义自定义基底列表 (Custom Basis)
    # ==========================================================================
    # 索引映射: A0->0, A1->1; B0->2, B1->3, B2->4

    custom_basis_list = [
        # --- 恒等算符 (自动包含，但显式写出也没问题) ---
        (),

        # --- 单体算符 ---
        (0,), (1,),  # E0, E1
        (2,), (3,), (4,),  # F0, F1, F2

        # --- Alice 高阶项 ---
        (0, 1), (1, 0),  # E0E1, E1E0
        (0, 1, 0),  # E0E1E0

        # --- Bob 高阶项 ---
        (2, 3), (3, 2),  # F0F1, F1F0
        (2, 4), (4, 2),  # F0F2, F2F0
        (3, 4), (4, 3),  # F1F2, F2F1
        (2, 4, 2)  # F0F2F0
    ]

    # 2. 构建 NPA 变量 (使用自定义基底)
    dummy_cg = np.zeros((3, 4))

    # 调用 custom 版本，传入 custom_basis
    G, constraints, ind_catalog = npa_constraints(
        dummy_cg,
        desc,
        custom_basis=custom_basis_list,  # 传入列表
        enforce_data=False
    )
    # ... 调用 npa_constraints 后 ...
    print(f"\n[Debug] Gamma Matrix Shape: {G.shape}")
    print(f"[Debug] Custom Basis Length: {len(custom_basis_list)}")

    # 验证是否只生成了预期的变量
    # 如果 shape[0] 远大于 len(custom_basis)+1，说明 NPAHierarchy_Custom 里的筛选逻辑没生效
    if G.shape[0] != len(custom_basis_list):
        print("[Warning] Gamma 矩阵维度与自定义基底长度不匹配！请检查筛选逻辑。")
    # print(f"[Info] Gamma 矩阵维度: {G.shape}")

    # 3. [Constraint 1] Bell 不等式约束
    bell_expr = build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix)
    constraints.append(bell_expr == target_val)

    # 4. [Constraint 2] Localizing Matrix 约束
    try:
        loc_constraints = build_localizing_matrix_constraint(G, ind_catalog, m_vec, alpha)
        constraints.extend(loc_constraints)
    except Exception as e:
        print(f"Localizing Matrix 构建失败: {e}")
        return None

    # 5. 构造 Fidelity 目标
    try:
        part_1, part_2, part_3 = build_fidelity_objective(G, ind_catalog, m_vec, theta)
        func = part_1 + part_2 + part_3
    except ValueError as e:
        print(f"目标函数构建失败: {e}")
        return None

    # 6. 求解 有改动
    print(f"Solving SDP for Target={target_val:.4f}...")
    # prob = cp.Problem(cp.Minimize(func), constraints)
    prob = cp.Problem(cp.Maximize(func), constraints)

    try:
        # 推荐使用 MOSEK，如果不行换 CVXOPT
        prob.solve(solver=cp.MOSEK, verbose=True)
    except cp.SolverError:
        return None

    # [调试打印]
    # print(f"Solver Status: {prob.status}")
    print(f"\n[Debug] Solver Status: {prob.status}")
    print(f"[Debug] Optimal Value (Fidelity): {prob.value}")

    if G.value is None:
        print("[Error] G.value is None. 求解失败。")
        return None

    # 检查归一化: <I> 必须为 1
    norm_val = G.value[0, 0]
    print(f"[Debug] Normalization G[0,0]: {norm_val:.6f}")
    if abs(norm_val - 1.0) > 1e-4:
        print("[CRITICAL] 归一化严重失效！所有结果都不可信。")

    p1_val = part_1.value
    p2_val = part_2.value
    p3_val = part_3.value

    print(f"[Debug] Fidelity Decomposition:")
    print(f"    Part 1 (CHSH part): {p1_val:.6f}")
    print(f"    Part 2 (Positive):  {p2_val:.6f}")
    print(f"    Part 3 (Negative):  {p3_val:.6f}")
    # 这里的求和应该等于 prob.value
    print(f"    Total Sum:          {p1_val + p2_val + p3_val:.6f}")


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

    # 1. 输入 Theta
    theta = np.pi / 6

    # 2. 根据 Theta 计算对应的 Alpha
    sin_2theta = np.sin(2 * theta)
    cos_2theta = np.cos(2 * theta)
    alpha_val = 2 * cos_2theta / np.sqrt(1 + sin_2theta ** 2)
    mu = np.arctan(sin_2theta)

    # # 3. 输入系数矩阵
    # vec_ineq2 = [alpha_val, 0, 0, 0, 1., 1., 1., -1.]
    # coeff_ineq2 = convert_vector_to_matrix(vec_ineq2)
    # 辅助变量
    cos_mu = np.cos(mu)
    cos_2mu = np.cos(2 * mu)  # 或者使用 localizing matrix 里的 k1, k2 相关逻辑

    # ---------------------------------------------------------
    # [关键修改] 使用“旋转后”的系数
    # 目的: 强迫 SDP 输出的 B0 算符对齐到 Z 轴
    # ---------------------------------------------------------
    vec_ineq_rotated = [
        alpha_val,  # A0 系数
        0,  # A1 系数
        0,  # B0 系数
        0,  # B1 系数
        2 * cos_mu,  # A0B0 系数
        0,  # A0B1 系数
        cos_2mu / cos_mu,  # A1B0 系数
        -1.0 / cos_mu  # A1B1 系数
    ]

    # 使用新系数构建矩阵
    coeff_ineq2 = convert_vector_to_matrix(vec_ineq_rotated)

    # 4. 计算边界
    quantum_bound = np.sqrt(8 + 2 * alpha_val ** 2)
    local_bound = calculate_classical_bound(coeff_ineq2)

    print(f"\n[Main] Theta={theta:.4f}, Derived Alpha={alpha_val:.4f}")
    print(f"Quantum Bound = {quantum_bound:.6f}")

    # 5. 测试 (不再需要传 k_level)
    fid_max = calculate_fidelity(quantum_bound, coeff_ineq2, desc, theta)

    if fid_max is not None:
        print(f"Min Fidelity: {fid_max:.8f}")
    else:
        print("Calculation Failed.")