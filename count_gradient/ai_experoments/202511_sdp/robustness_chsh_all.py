import numpy as np
import cvxpy as cp
import sys

# 导入依赖
try:
    from NPAHierarchy_3 import npa_constraints
    from locate_gamma_2 import get_gamma_element
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


# ==============================================================================
# 1. 工具函数：系数转换与经典界计算
# ==============================================================================
def convert_vector_to_matrix(vec):
    """
    将 8 元素向量转换为 3x3 Full Correlation 系数矩阵。
    Vector: [A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1]
    M_fc:
        [K,  B0,   B1]
        [A0, A0B0, A0B1]
        [A1, A1B0, A1B1]
    """
    if len(vec) != 8:
        raise ValueError("系数向量必须包含 8 个元素。")

    A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1 = vec
    M = np.zeros((3, 3))
    M[0, 1], M[0, 2] = B0, B1
    M[1, 0], M[1, 1], M[1, 2] = A0, A0B0, A0B1
    M[2, 0], M[2, 1], M[2, 2] = A1, A1B0, A1B1
    return M


def calculate_classical_bound(coeff_matrix):
    """
    遍历所有确定性策略，计算给定不等式的经典界 (Local Bound)。
    策略: Ax, By \in {+1, -1}
    """
    max_val = -np.inf
    # 遍历 A0, A1, B0, B1 的所有 16 种组合
    for a0 in [-1, 1]:
        for a1 in [-1, 1]:
            for b0 in [-1, 1]:
                for b1 in [-1, 1]:
                    # 计算当前策略下的不等式值
                    val = 0
                    val += coeff_matrix[0, 0]  # K (虽然这里通常为0)

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
# 2. 约束构建器：通用 Bell 不等式 (由系数矩阵决定)
# ==============================================================================
def build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix):
    """
    根据输入的系数矩阵，构建用于 SDP 约束的 Bell 表达式。
    expr = sum( coeff * expectation_from_G )
    """
    expr = 0
    if coeff_matrix[0, 0] != 0:
        expr += coeff_matrix[0, 0]

    def get_term_E(alice_list, bob_list):
        t = get_gamma_element(G, ind_catalog, m_vec, alice_list, bob_list)
        if t is None:
            raise ValueError(f"约束构建失败: 无法在 Gamma 矩阵中找到算符 A{alice_list}B{bob_list}。")
        return t

    # Bob 单体项
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
                # <AxBy> = 4<ExFy> - 2<Ex> - 2<Fy> + 1
                expr += c * (4 * term_ExFy - 2 * term_Ex - 2 * term_Fy + 1)
    return expr


# ==============================================================================
# 3. 目标函数构建器：固定为标准 CHSH 形式的 Fidelity
# ==============================================================================
def build_fidelity_objective(G, ind_catalog, m_vec):
    """
    构建 Fidelity 目标函数。
    """

    def get_term(alice_seq, bob_seq):
        val = get_gamma_element(G, ind_catalog, m_vec, alice_seq, bob_seq)
        if val is None:
            raise ValueError(f"目标函数构建失败: Term <A{alice_seq} B{bob_seq}> not found.")
        return val

    # --- 基础定义 ---
    A0, A1 = 0, 1
    B0, B1 = 2, 3
    I = []

    # --- (A) 纯单边项 (Pure Local Terms) ---
    term_I = get_term(I, I)

    # Alice Singles & Strings
    term_E0 = get_term([A0], I)
    term_E1 = get_term([A1], I)
    term_E0E1 = get_term([A0, A1], I)
    term_E1E0 = get_term([A1, A0], I)
    term_E0E1E0 = get_term([A0, A1, A0], I)  # Length 3

    # Bob Singles & Strings
    term_F0 = get_term(I, [B0])
    term_F1 = get_term(I, [B1])
    term_F0F1 = get_term(I, [B0, B1])
    term_F1F0 = get_term(I, [B1, B0])
    term_F0F1F0 = get_term(I, [B0, B1, B0])  # Length 3

    # --- (B) 1-1 交叉项 (Cross Terms 1-1) ---
    term_E0_F0 = get_term([A0], [B0])
    term_E0_F1 = get_term([A0], [B1])
    term_E1_F0 = get_term([A1], [B0])
    term_E1_F1 = get_term([A1], [B1])

    # --- (C) 2-1 和 1-2 交叉项 (Cross Terms 2-1, 1-2) ---
    # Alice 2, Bob 1
    term_E0E1_F0 = get_term([A0, A1], [B0])
    term_E0E1_F1 = get_term([A0, A1], [B1])
    term_E1E0_F0 = get_term([A1, A0], [B0])
    term_E1E0_F1 = get_term([A1, A0], [B1])

    # Alice 1, Bob 2
    term_E0_F0F1 = get_term([A0], [B0, B1])
    term_E0_F1F0 = get_term([A0], [B1, B0])
    term_E1_F0F1 = get_term([A1], [B0, B1])
    term_E1_F1F0 = get_term([A1], [B1, B0])

    # --- (D) 3-1 和 1-3 交叉项 (Cross Terms 3-1, 1-3) ---
    # Alice 3, Bob 1
    term_E0E1E0_F0 = get_term([A0, A1, A0], [B0])
    term_E0E1E0_F1 = get_term([A0, A1, A0], [B1])

    # Alice 1, Bob 3
    term_E0_F0F1F0 = get_term([A0], [B0, B1, B0])
    term_E1_F0F1F0 = get_term([A1], [B0, B1, B0])

    # --- (E) 2-2 交叉项 (Cross Terms 2-2) ---
    term_E0E1_F0F1 = get_term([A0, A1], [B0, B1])
    term_E0E1_F1F0 = get_term([A0, A1], [B1, B0])
    term_E1E0_F0F1 = get_term([A1, A0], [B0, B1])
    term_E1E0_F1F0 = get_term([A1, A0], [B1, B0])

    # --- (F) 3-2 和 2-3 交叉项 (用于 6-body 展开) ---
    term_E0E1E0_F0F1 = get_term([A0, A1, A0], [B0, B1])
    term_E0E1E0_F1F0 = get_term([A0, A1, A0], [B1, B0])

    term_E0E1_F0F1F0 = get_term([A0, A1], [B0, B1, B0])
    term_E1E0_F0F1F0 = get_term([A1, A0], [B0, B1, B0])

    # --- (G) 3-3 交叉项 (最高阶) ---
    term_E0E1E0_F0F1F0 = get_term([A0, A1, A0], [B0, B1, B0])

    # ==============================================================================
    # 6. 组装目标函数各部分 (d_Terms)
    # ==============================================================================

    # --- 2-body terms ---
    d_A0B0 = 4 * term_E0_F0 - 2 * term_E0 - 2 * term_F0 + term_I
    d_A1B1 = 4 * term_E1_F1 - 2 * term_E1 - 2 * term_F1 + term_I
    d_A0B1 = 4 * term_E0_F1 - 2 * term_E0 - 2 * term_F1 + term_I
    d_A1B0 = 4 * term_E1_F0 - 2 * term_E1 - 2 * term_F0 + term_I

    # 显式计算 Standard CHSH Value
    d_CHSH_std = d_A0B0 + d_A0B1 + d_A1B0 - d_A1B1

    # --- 4-body terms (Mixed 3-1 split) ---
    d_A0A1A0B1 = (16 * term_E0E1E0_F1
                  - 8 * (term_E0E1_F1 + term_E1E0_F1 + term_E0E1E0)
                  + 4 * (term_E0E1 + term_E1E0 + term_E1_F1)
                  - 2 * (term_E1 + term_F1)
                  + term_I)

    d_A0A1A0B0 = (16 * term_E0E1E0_F0
                  - 8 * (term_E0E1_F0 + term_E1E0_F0 + term_E0E1E0)
                  + 4 * (term_E0E1 + term_E1E0 + term_E1_F0)
                  - 2 * (term_E1 + term_F0)
                  + term_I)

    d_A1B0B1B0 = (16 * term_E1_F0F1F0
                  - 8 * (term_F0F1F0 + term_E1_F0F1 + term_E1_F1F0)
                  + 4 * (term_E1_F1 + term_F0F1 + term_F1F0)
                  - 2 * (term_E1 + term_F1)
                  + term_I)

    d_A0B0B1B0 = (16 * term_E0_F0F1F0
                  - 8 * (term_F0F1F0 + term_E0_F0F1 + term_E0_F1F0)
                  + 4 * (term_E0_F1 + term_F0F1 + term_F1F0)
                  - 2 * (term_E0 + term_F1)
                  + term_I)

    # --- 4-body terms (Mixed 2-2 split) ---
    d_A0A1B0B1 = (16 * term_E0E1_F0F1
                  - 8 * (term_E0E1_F0 + term_E0E1_F1 + term_E0_F0F1 + term_E1_F0F1)
                  + 4 * (term_E0_F0 + term_E0_F1 + term_E1_F0 + term_E1_F1)
                  + 4 * (term_E0E1 + term_F0F1)
                  - 2 * (term_E0 + term_E1 + term_F0 + term_F1)
                  + term_I)

    d_A0A1B1B0 = (16 * term_E0E1_F1F0
                  - 8 * (term_E0E1_F1 + term_E0E1_F0 + term_E0_F1F0 + term_E1_F1F0)
                  + 4 * (term_E0_F1 + term_E0_F0 + term_E1_F1 + term_E1_F0)
                  + 4 * (term_E0E1 + term_F1F0)
                  - 2 * (term_E0 + term_E1 + term_F1 + term_F0)
                  + term_I)

    d_A1A0B0B1 = (16 * term_E1E0_F0F1
                  - 8 * (term_E1E0_F0 + term_E1E0_F1 + term_E1_F0F1 + term_E0_F0F1)
                  + 4 * (term_E1_F0 + term_E1_F1 + term_E0_F0 + term_E0_F1)
                  + 4 * (term_E1E0 + term_F0F1)
                  - 2 * (term_E1 + term_E0 + term_F0 + term_F1)
                  + term_I)

    d_A1A0B1B0 = (16 * term_E1E0_F1F0
                  - 8 * (term_E1E0_F1 + term_E1E0_F0 + term_E1_F1F0 + term_E0_F1F0)
                  + 4 * (term_E1_F1 + term_E1_F0 + term_E0_F1 + term_E0_F0)
                  + 4 * (term_E1E0 + term_F1F0)
                  - 2 * (term_E1 + term_E0 + term_F1 + term_F0)
                  + term_I)

    d_A0A1A0B0B1B0 = (64 * term_E0E1E0_F0F1F0
                      - 32 * (term_E0E1E0_F0F1 + term_E0E1E0_F1F0 + term_E0E1_F0F1F0 + term_E1E0_F0F1F0)
                      + 16 * (
                              term_E0E1E0_F1 + term_E1_F0F1F0 + term_E0E1_F0F1 + term_E0E1_F1F0 + term_E1E0_F0F1 + term_E1E0_F1F0)
                      - 8 * (term_E0E1E0 + term_F0F1F0 + term_E0E1_F1 + term_E1E0_F1 + term_E1_F0F1 + term_E1_F1F0)
                      + 4 * (term_E0E1 + term_E1E0 + term_F0F1 + term_F1F0 + term_E1_F1)
                      - 2 * (term_E1 + term_F1)
                      + term_I)

    # --- 最终目标函数 ---
    part_1 = 0.5 + d_CHSH_std / (2 * np.sqrt(2))
    part_2 = - (1 / 8) * (d_A0A1B0B1 - d_A0A1B1B0 - d_A1A0B0B1 + d_A1A0B1B0)
    part_3 = (1 / (8 * np.sqrt(2))) * (
            3 * d_A1B1 - 2 * d_A0B1 - 2 * d_A1B0 + d_A0A1A0B1 - 2 * d_A0A1A0B0 + d_A1B0B1B0 - 2 * d_A0B0B1B0 - d_A0A1A0B0B1B0
    )

    return (part_1 + part_2 + part_3) / 2


# ==============================================================================
# 4. 主计算函数
# ==============================================================================
def calculate_fidelity(target_val: float, coeff_matrix: np.ndarray, desc: np.ndarray, k: int = 3) -> float:
    m_vec = desc[2:4]

    # 1. 哑数据构建变量
    dummy_cg = np.zeros((3, 3))
    G, constraints, ind_catalog = npa_constraints(dummy_cg, desc, k=k, enforce_data=False)

    # 2. 施加约束: 输入的不等式(由coeff_matrix定义) == target_val
    bell_expr = build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix)
    constraints.append(bell_expr == target_val)

    # 3. 构造 Fidelity 目标 (不依赖 target_val，只依赖 G)
    func = build_fidelity_objective(G, ind_catalog, m_vec)

    # 4. 求解
    prob = cp.Problem(cp.Minimize(func), constraints)
    try:
        # 提高精度，增加迭代
        prob.solve(solver=cp.SCS, eps=1e-7, max_iters=20000)
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
    desc = np.array([2, 2, 2, 2])
    k_level = 3

    # --- 输入系数矩阵 2 ---
    # Vector: [A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1]
    vec_ineq2 = [0., 0.565685, -0.4, 0.4, 1., 1., 1., -1.]
    coeff_ineq2 = convert_vector_to_matrix(vec_ineq2)

    # --- 1. 自动计算经典界 (Classical Bound) ---
    print("\n[Analysis] Calculating Classical Bound...")
    local_bound = calculate_classical_bound(coeff_ineq2)
    print(f"Classical Bound = {local_bound:.6f}")

    # --- 2. 设定最大量子违背 (Quantum Bound) ---
    # 你提到它也是 2sqrt(2)
    quantum_bound = 2 * np.sqrt(2)
    print(f"Quantum Bound   = {quantum_bound:.6f}")

    # --- 3. 测试 1: 最大量子违背 ---
    print(f"\n=== Test Case 1: Target Violation = Quantum Bound ({quantum_bound:.4f}) ===")
    fid_max = calculate_fidelity(quantum_bound, coeff_ineq2, desc, k=k_level)
    if fid_max is not None:
        print(f"Min Fidelity: {fid_max:.8f}")
    else:
        print("Calculation Failed.")

    # --- 4. 测试 2: 手动设定的中间值 (例如 2.5) ---
    test_val = 2.6
    print(f"\n=== Test Case 2: Target Violation = {test_val:.4f} ===")
    fid_mid = calculate_fidelity(test_val, coeff_ineq2, desc, k=k_level)
    if fid_mid is not None:
        print(f"Min Fidelity: {fid_mid:.8f}")
    else:
        print("Calculation Failed.")

    # --- 4. 测试 3: 经典界 ---
    print(f"\n=== Test Case 3: Target Violation = Local Bound ({local_bound:.4f}) ===")
    fid_mid = calculate_fidelity(local_bound, coeff_ineq2, desc, k=k_level)
    if fid_mid is not None:
        print(f"Min Fidelity: {fid_mid:.8f}")
    else:
        print("Calculation Failed.")