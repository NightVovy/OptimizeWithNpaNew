import numpy as np
import cvxpy as cp
import sys

# 导入依赖
try:
    from NPAHierarchy_3 import npa_constraints
    from locate_gamma_2 import get_gamma_element
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 NPAHierarchy_3.py, locate_gamma_2.py, npa_support.py 都在当前目录下。")
    sys.exit(1)


# ==============================================================================
# 1. 通用 Bell 表达式构建器 (支持任意 3x3 系数矩阵)
# ==============================================================================
def build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix):
    """
    根据 3x3 系数矩阵构建 CVXPY 表达式。
    coeff_matrix 结构:
        [ K,   B0,   B1   ]
        [ A0,  A0B0, A0B1 ]
        [ A1,  A1B0, A1B1 ]
    """
    expr = 0

    # 1. 常数项 (K)
    if coeff_matrix[0, 0] != 0:
        expr += coeff_matrix[0, 0]

    # 定义辅助查找函数
    def get_term_E(alice_list, bob_list):
        # 查找 <E... F...>。注意：m_vec 必须传递
        t = get_gamma_element(G, ind_catalog, m_vec, alice_list, bob_list)
        if t is None:
            raise ValueError(f"无法在 Gamma 矩阵中找到算符 A{alice_list}B{bob_list}。请增大 K。")
        return t

    # 2. Bob 单体项 <By> -> coeff * (2<Fy> - 1)
    # y_idx: 0->B0, 1->B1
    for y_idx in range(2):
        c = coeff_matrix[0, y_idx + 1]
        if c != 0:
            # Bob 设置索引: 2 + y_idx (因为 Alice 有 2 个设置)
            bob_setting = [2 + y_idx]
            term_Fy = get_term_E([], bob_setting)
            expr += c * (2 * term_Fy - 1)

    # 3. Alice 单体项 <Ax> -> coeff * (2<Ex> - 1)
    # x_idx: 0->A0, 1->A1
    for x_idx in range(2):
        c = coeff_matrix[x_idx + 1, 0]
        if c != 0:
            term_Ex = get_term_E([x_idx], [])
            expr += c * (2 * term_Ex - 1)

    # 4. 关联项 <AxBy> -> coeff * (4<ExFy> - 2<Ex> - 2<Fy> + 1)
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
# 2. Fidelity 目标函数构建器 (封装了那一大串公式)
# ==============================================================================
def build_fidelity_objective(G, ind_catalog, m_vec, chsh_val_for_func):
    """
    构建基于 SWAP 方法的 Fidelity 目标函数。
    """

    # 内部辅助函数：简化 get_gamma_element 调用
    def get_term(alice_seq, bob_seq):
        val = get_gamma_element(G, ind_catalog, m_vec, alice_seq, bob_seq)
        if val is None:
            raise ValueError(f"Term <A{alice_seq} B{bob_seq}> not found.")
        return val

    # --- 索引定义 ---
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
    d_A1B1 = 4 * term_E1_F1 - 2 * term_E1 - 2 * term_F1 + term_I
    d_A0B1 = 4 * term_E0_F1 - 2 * term_E0 - 2 * term_F1 + term_I
    d_A1B0 = 4 * term_E1_F0 - 2 * term_E1 - 2 * term_F0 + term_I

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

    # --- 6-body term ---
    d_A0A1A0B0B1B0 = (64 * term_E0E1E0_F0F1F0
                      - 32 * (term_E0E1E0_F0F1 + term_E0E1E0_F1F0 + term_E0E1_F0F1F0 + term_E1E0_F0F1F0)
                      + 16 * (
                              term_E0E1E0_F1 + term_E1_F0F1F0 + term_E0E1_F0F1 + term_E0E1_F1F0 + term_E1E0_F0F1 + term_E1E0_F1F0)
                      - 8 * (term_E0E1E0 + term_F0F1F0 + term_E0E1_F1 + term_E1E0_F1 + term_E1_F0F1 + term_E1_F1F0)
                      + 4 * (term_E0E1 + term_E1E0 + term_F0F1 + term_F1F0 + term_E1_F1)
                      - 2 * (term_E1 + term_F1)
                      + term_I)

    # --- 最终目标函数 ---
    part_1 = 0.5 + chsh_val_for_func / (2 * np.sqrt(2))
    part_2 = - (1 / 8) * (d_A0A1B0B1 - d_A0A1B1B0 - d_A1A0B0B1 + d_A1A0B1B0)
    part_3 = (1 / (8 * np.sqrt(2))) * (
            3 * d_A1B1 - 2 * d_A0B1 - 2 * d_A1B0 + d_A0A1A0B1 - 2 * d_A0A1A0B0 + d_A1B0B1B0 - 2 * d_A0B0B1B0 - d_A0A1A0B0B1B0
    )

    return (part_1 + part_2 + part_3) / 2


# ==============================================================================
# 3. 主计算函数
# ==============================================================================
def robustness_chsh(coeff_matrix: np.ndarray, target_val: float, desc: np.ndarray, k: int = 3) -> float:
    """
    计算给定不等式系数 coeff_matrix 在违背值 target_val 下的 Fidelity 下界。
    """
    m_vec = desc[2:4]

    # 1. 哑数据 (Dummy)，因为我们 set enforce_data=False
    dummy_cg = np.zeros((3, 3))

    # 2. 构建 NPA 变量 (数据松绑)
    G, constraints, ind_catalog = npa_constraints(dummy_cg, desc, k=k, enforce_data=False)

    # 3. 施加不等式约束: Bell_Expr(G) == target_val
    bell_expr = build_general_bell_expression(G, ind_catalog, m_vec, coeff_matrix)
    constraints.append(bell_expr == target_val)

    # 4. 构建目标函数
    # 注意: 这里的 target_val 传给 func，意味着 func 内部的 chsh_val = target_val
    func = build_fidelity_objective(G, ind_catalog, m_vec, target_val)

    # 5. 求解 (Minimize Fidelity)
    print(f"Solving SDP for Target={target_val:.4f}...")
    prob = cp.Problem(cp.Minimize(func), constraints)

    try:
        # 使用较高的精度，增加迭代次数
        prob.solve(solver=cp.SCS, eps=1e-7, max_iters=20000)
    except cp.SolverError as e:
        print(f"Solver Error: {e}")
        return None

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return prob.value
    else:
        print(f"Optimization Status: {prob.status}")
        return None


# ==============================================================================
# 4. 运行示例
# ==============================================================================
if __name__ == '__main__':
    # 设置
    desc = np.array([2, 2, 2, 2])
    k_level = 3

    # --- 不等式 1: 标准 CHSH ---
    # Coeffs: A0B0 + A0B1 + A1B0 - A1B1
    coeff_chsh = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, -1]
    ])

    print("\n=== Robustness Check: Standard CHSH ===")
    # 测试两个点：最大违背和稍微小一点的值
    for val in [2.8284, 2]:
        fid = robustness_chsh(coeff_chsh, val, desc, k=k_level)
        if fid is not None:
            print(f"Violation {val:.4f} -> Min Fidelity: {fid:.6f}")

