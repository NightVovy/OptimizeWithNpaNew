import numpy as np
import cvxpy as cp
import sys

# 导入依赖模块
try:
    from fc2cg import FC2CG
    from NPAHierarchy_2 import npa_constraints
    from locate_gamma_2 import get_gamma_element
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


def fidelity_chsh(chsh_val: float, M_fc: np.ndarray, desc: np.ndarray, k: int = 3) -> float:
    """
    构造并求解 SDP 以计算 Bell 不等式的 Fidelity。
    目标函数包含复杂的 2-body, 4-body 和 6-body 关联项。
    """

    # 1. 基础设置
    m_vec = desc[2:4]  # [2, 2]
    cg_matrix = FC2CG(M_fc, behaviour=1)

    # 2. 调用构建器获取变量(G)和约束
    # 注意：为了包含 E0E1E0 (长度3) 和 F0F1F0 (长度3) 的组合，k 必须至少为 3
    G, constraints, ind_catalog, *rest = npa_constraints(cg_matrix, desc, k=k)

    # --- 3. 定义便利函数 (Helper Wrapper) ---
    def get_term(alice_seq, bob_seq):
        val = get_gamma_element(G, ind_catalog, m_vec, alice_seq, bob_seq)
        if val is None:
            # 如果找不到，抛出详细错误提示用户增大 k
            raise ValueError(f"项 <A{alice_seq} B{bob_seq}> 在 Gamma 矩阵(k={k})中未找到。")
        return val

    # --- 4. 定义基础索引 ---
    # Alice Settings
    A0, A1 = 0, 1
    # Bob Settings (Offset by MA=2)
    B0, B1 = 2, 3
    # Identity (Empty List)
    I = []

    # ==============================================================================
    # 5. 获取所有需要的 Gamma 矩阵元素 (Terms)
    #    命名约定: term_AlicePart_BobPart (若某方为 I 则省略)
    # ==============================================================================

    print("Constructing Terms...")

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

    # ==============================================================================
    # 7. 定义最终目标函数 func
    # ==============================================================================
    # func = 1/2 + CHSH * 1/(2*np.sqrt(2))
    #        - 1/8 * (d_A0A1B0B1 - d_A0A1B1B0 - d_A1A0B0B1 + d_A1A0B1B0)
    #        + 1/(8*np.sqrt(2)) * (3 * d_A1B1 - 2 * d_A0B1 - 2 * d_A1B0
    #           + d_A0A1A0B1 - 2 * d_A0A1A0B0 + d_A1B0B1B0 - 2 * d_A0B0B1B0
    #           - d_A0A1A0B0B1B0)

    part_1 = 0.5 + chsh_val / (2 * np.sqrt(2))

    part_2 = - (1 / 8) * (d_A0A1B0B1 - d_A0A1B1B0 - d_A1A0B0B1 + d_A1A0B1B0)

    part_3 = (1 / (8 * np.sqrt(2))) * (
            3 * d_A1B1
            - 2 * d_A0B1
            - 2 * d_A1B0
            + d_A0A1A0B1
            - 2 * d_A0A1A0B0
            + d_A1B0B1B0
            - 2 * d_A0B0B1B0
            - d_A0A1A0B0B1B0
    )

    func = part_1 + part_2 + part_3

    # --- 8. 求解 ---
    print(f"Solving SDP with CHSH={chsh_val:.4f}, Level k={k}...")
    prob = cp.Problem(cp.Minimize(func), constraints)

    try:
        # 复杂算符可能需要较高精度
        prob.solve(solver=cp.SCS, eps=1e-7, max_iters=10000)
    except cp.SolverError as e:
        print(f"Solver Error: {e}")
        return None

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return prob.value
    else:
        print(f"Optimization failed. Status: {prob.status}")
        return None


# --- 主程序入口 ---
if __name__ == '__main__':
    # 1. 准备数据
    print("Preparing Data...")
    a0 = 0
    a1 = 0
    b0 = 0
    b1 = 0
    e = 1 / np.sqrt(2)

    # M_fc: Full Correlation 矩阵
    M_fc_input = np.array([
        [0.0, b0, b1],
        [a0, e, e],
        [a1, e, -e]
    ])

    # 描述向量 [OA, OB, MA, MB]
    desc_input = np.array([2, 2, 2, 2])

    # 测试 Tsirelson Bound 值 2 * np.sqrt(2)
    test_val = 2

    # 2. 运行函数
    # 注意: 包含长度为 3 的算符 (A0 A1 A0)，所以 k 必须至少是 3
    result = fidelity_chsh(test_val, M_fc_input, desc_input, k=3)

    if result is not None:
        print(f"\nFinal Result (Minimized func): {result:.8f}")
    else:
        print("\nResult: None")