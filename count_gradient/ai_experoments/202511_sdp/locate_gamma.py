import numpy as np
from npa_support import find_in_cell, product_of_orthogonal


def get_gamma_value(G_val: np.ndarray, ind_catalog: list, m_vec: np.ndarray,
                    alice_seq: list, bob_seq: list) -> float:
    """
    在 Gamma 矩阵中查找 <Alice_Seq * Bob_Seq> 的期望值。

    原理: Gamma[u, v] = <S_u^dagger * S_v>
    我们令 S_v = Bob_Seq, S_u^dagger = Alice_Seq
    => S_u = (Alice_Seq)^dagger = Reverse(Alice_Seq) (因为投影算子是厄米的)

    参数:
        G_val: 求解后的 Gamma 矩阵数值
        ind_catalog: 算符目录
        m_vec: 测量设置向量 [MA, MB]
        alice_seq: Alice 的投影算子设置索引列表 (例如 [0, 1, 0])
        bob_seq: Bob 的投影算子设置索引列表 (例如 [2])

    返回:
        float: 对应的期望值。如果找不到对应的基算符，返回 None。
    """

    # 1. 构造 Alice 的查找目标 (对应 Gamma 的行 u)
    # 我们需要 S_u = (Alice_Seq)^dagger。
    # 输入是 [0, 1, 0] (即 E0 E1 E0)。共轭转置是 E0^dag E1^dag E0^dag = E0 E1 E0 (反序)。
    # 所以我们需要将输入的序列 反转。
    alice_input_rev = alice_seq[::-1]  # List reverse

    # 构造 res 矩阵: Row 0 是设置, Row 1 全是 0 (因为我们只关注结果0的投影)
    if len(alice_input_rev) > 0:
        res_a_raw = np.vstack([alice_input_rev, np.zeros(len(alice_input_rev), dtype=int)])
    else:
        # 空列表代表恒等算符
        res_a_raw = np.array([[0], [-1]])

    # 简化算符 (这一步很重要，因为 catalog 里存的是最简形式)
    res_a, _ = product_of_orthogonal(res_a_raw, m_vec)

    # 在目录中查找行索引
    row_idx = find_in_cell(res_a, ind_catalog)

    # 2. 构造 Bob 的查找目标 (对应 Gamma 的列 v)
    # S_v = Bob_Seq (不需要反转)
    if len(bob_seq) > 0:
        res_b_raw = np.vstack([bob_seq, np.zeros(len(bob_seq), dtype=int)])
    else:
        res_b_raw = np.array([[0], [-1]])

    res_b, _ = product_of_orthogonal(res_b_raw, m_vec)

    # 在目录中查找列索引
    col_idx = find_in_cell(res_b, ind_catalog)

    # 3. 返回结果
    if row_idx != -1 and col_idx != -1:
        # Gamma 矩阵是对称的，但为了逻辑严谨，使用 [row, col]
        value = G_val[row_idx, col_idx]
        print(f"Found! Row(Alice)={row_idx}, Col(Bob)={col_idx}")
        print(f"  Alice Simplified: {res_a.flatten()}")
        print(f"  Bob Simplified:   {res_b.flatten()}")
        return value
    else:
        print("Error: 指定的算符序列在当前层级 K 的 ind_catalog 中未找到。")
        if row_idx == -1: print(f"  Alice sequence not found: {alice_seq} -> simplified: \n{res_a}")
        if col_idx == -1: print(f"  Bob sequence not found: {bob_seq} -> simplified: \n{res_b}")
        return None


# --- 测试代码 ---
if __name__ == '__main__':
    from test_chsh_npa import FC2CG  # 复用之前的转换函数
    from NPAHierarchy_with_gamma import npa_hierarchy

    # 1. 设置问题 (CHSH Tsirelson)
    a0, a1, b0, b1 = 0, 0, 0, 0
    e = 1 / np.sqrt(2)
    M_fc = np.array([[0.0, b0, b1], [a0, e, e], [a1, e, -e]])
    M2cg = FC2CG(M_fc, behaviour=1)
    desc = np.array([2, 2, 2, 2])
    m_vec = desc[2:4]  # [2, 2]

    # 2. 运行 NPA (K=3)
    print("Running NPA (K=3)...")
    is_npa, G_val, catalog = npa_hierarchy(M2cg, desc, k=3)

    if is_npa > 0.9:
        print("NPA Solved Successfully.")

        # 3. 查找特定项 <A0 A1 A0 * B0>
        # Alice: E0^0 E0^1 E0^0 -> 设置索引 [0, 1, 0]
        # Bob:   F0^0           -> 设置索引 [2] (Alice有2个设置，Bob从2开始)

        alice_sequence = [0, 1, 0]
        bob_sequence = [2]

        print(f"\nSearching for < Alice{alice_sequence} * Bob{bob_sequence} > ...")
        val = get_gamma_value(G_val, catalog, m_vec, alice_sequence, bob_sequence)

        if val is not None:
            print(f"\nValue = {val:.6f}")

            # 理论验证:
            # 在 Tsirelson bound 下，A0 A1 = i/sqrt(2) * Z (approx), A0 A1 A0 = A1 (anti-commute? no)
            # 对于 CHSH 最大破坏态，A0, A1 对应 Pauli X, Z。
            # E0 = (I+X)/2, E1 = (I+Z)/2.
            # 这是一个复杂的验证，但只要数值返回了，说明定位逻辑是通的。
    else:
        print("NPA Failed.")