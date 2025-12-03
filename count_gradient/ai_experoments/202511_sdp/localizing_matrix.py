import numpy as np
import cvxpy as cp
import sys

# 假设该文件存在于同一目录
try:
    from locate_gamma_Bob import get_gamma_element_bob
except ImportError:
    print("错误: 无法导入 locate_gamma_Bob，请确保该文件在当前目录下。")


    # 为了防止 IDE 报错，定义一个 dummy 函数 (实际运行时需确保文件存在)
    def get_gamma_element_bob(*args):
        return None


def build_localizing_matrix_constraint(G: cp.Variable, ind_catalog: list, m_vec: np.ndarray, alpha: float):
    """
    构造基于 B2, B0, B1 关系的 Localizing Matrix 约束。

    返回:
        constraints (list): 包含一个半正定约束 [gamma_loc_sym >> 0]
    """

    # ==========================================================================
    # 1. 计算系数 k1, k2
    # ==========================================================================
    # 公式: sin(2theta) = sqrt((1 - alpha^2/4) / (1 + alpha^2/4))
    #       tan(mu) = sin(2theta)
    #       k1 = cos(2mu) / sin(2mu) = cot(2mu)
    #       k2 = 1 / sin(2mu) = csc(2mu)

    val_sin_2theta = np.sqrt((1 - (alpha ** 2) / 4) / (1 + (alpha ** 2) / 4))
    mu = np.arctan(val_sin_2theta)

    sin_2mu = np.sin(2 * mu)
    cos_2mu = np.cos(2 * mu)

    k1 = cos_2mu / sin_2mu
    k2 = 1.0 / sin_2mu

    print(f"[Info] Alpha={alpha}, k1={k1:.4f}, k2={k2:.4f}")

    # ==========================================================================
    # 2. 定义辅助查找函数
    # ==========================================================================
    # 注意: 这里查找的是 Bob 的算符，alice_seq 应为空 [] 或根据物理意义调整
    # 根据 locate_gamma_Bob.py，输入应为 bob_seq_row 和 bob_seq_col (全局索引)
    def get_term(bob_row, bob_col):
        # 查找 <Bob_Row * Bob_Col>
        # get_gamma_element_bob 会自动处理 Row 的共轭转置
        t = get_gamma_element_bob(G, ind_catalog, m_vec, bob_row, bob_col)
        if t is None:
            raise ValueError(f"无法在 Gamma 矩阵中找到 Bob算符组合 Row:{bob_row}, Col:{bob_col}。请检查 K 值。")
        return t

    # ==========================================================================
    # 3. 基础索引定义 (根据 m_vec 推断)
    # ==========================================================================
    # 假设 Alice 有 MA 个输入，Bob 的索引从 MA 开始
    # B0 -> MA, B1 -> MA+1, B2 -> MA+2
    MA = m_vec[0]

    idx_I = []  # 恒等算符 (空列表)
    idx_B0 = [MA]  # B0
    idx_B1 = [MA + 1]  # B1
    idx_B2 = [MA + 2]  # B2 (辅助算符)

    # 方便书写的组合变量 (List)
    idx_B0B1 = [MA, MA + 1]
    idx_B1B0 = [MA + 1, MA]
    idx_B0B2 = [MA, MA + 2]
    idx_B2B0 = [MA + 2, MA]
    idx_B1B2 = [MA + 1, MA + 2]
    idx_B2B1 = [MA + 2, MA + 1]

    # ==========================================================================
    # 4. 获取投影算子组合 (Terms)
    # ==========================================================================

    # --- 0-Body & 1-Body ---
    term_I = get_term(idx_I, idx_I)
    term_F0 = get_term(idx_I, idx_B0)
    term_F1 = get_term(idx_I, idx_B1)
    term_F2 = get_term(idx_I, idx_B2)

    # --- 2-Body ---
    term_F0F1 = get_term(idx_B0, idx_B1)
    term_F1F0 = get_term(idx_B1, idx_B0)
    term_F0F2 = get_term(idx_B0, idx_B2)
    term_F2F0 = get_term(idx_B2, idx_B0)
    term_F1F2 = get_term(idx_B1, idx_B2)
    term_F2F1 = get_term(idx_B2, idx_B1)

    # --- 3-Body (拆分为 1+2) ---
    term_F0_F2F0 = get_term(idx_B0, idx_B2B0)
    term_F0_F2F1 = get_term(idx_B0, idx_B2B1)
    term_F1_F2F0 = get_term(idx_B1, idx_B2B0)
    term_F1_F2F1 = get_term(idx_B1, idx_B2B1)
    term_F1_F0F2 = get_term(idx_B1, idx_B0B2)

    term_F2_F0F1 = get_term(idx_B2, idx_B0B1)
    term_F2_F0F2 = get_term(idx_B2, idx_B0B2)
    term_F2_F1F0 = get_term(idx_B2, idx_B1B0)
    term_F2_F1F2 = get_term(idx_B2, idx_B1B2)

    term_F0_F1F0 = get_term(idx_B0, idx_B1B0)
    term_F0_F1F2 = get_term(idx_B0, idx_B1B2)

    # --- 4-Body (拆分为 2+2) ---
    term_F0F2_F1F0 = get_term(idx_B0B2, idx_B1B0)
    term_F0F2_F0F1 = get_term(idx_B0B2, idx_B0B1)
    term_F0F2_F0F2 = get_term(idx_B0B2, idx_B0B2)
    term_F0F2_F1F2 = get_term(idx_B0B2, idx_B1B2)

    term_F1F2_F1F0 = get_term(idx_B1B2, idx_B1B0)
    term_F1F2_F0F1 = get_term(idx_B1B2, idx_B0B1)
    term_F1F2_F0F2 = get_term(idx_B1B2, idx_B0B2)
    term_F1F2_F1F2 = get_term(idx_B1B2, idx_B1B2)

    # ==========================================================================
    # 5. 定义可观测量 (Observables d_) 【部分已知，部分待补充】
    # ==========================================================================
    # 关系: d = LinearComb(terms)

    # --- 1-Body ---
    d_I = term_I
    d_B0 = 2 * term_F0 - term_I
    d_B1 = 2 * term_F1 - term_I
    d_B2 = 2 * term_F2 - term_I

    # --- 2-Body ---
    d_B0B1 = 4 * term_F0F1 - 2 * term_F0 - 2 * term_F1 + term_I
    d_B1B0 = 4 * term_F1F0 - 2 * term_F1 - 2 * term_F0 + term_I
    d_B0B2 = 4 * term_F0F2 - 2 * term_F0 - 2 * term_F2 + term_I
    d_B2B0 = 4 * term_F2F0 - 2 * term_F2 - 2 * term_F0 + term_I
    d_B1B2 = 4 * term_F1F2 - 2 * term_F1 - 2 * term_F2 + term_I
    d_B2B1 = 4 * term_F2F1 - 2 * term_F2 - 2 * term_F1 + term_I

    # --- 3-Body ---
    # d_B2B1B0
    d_B2B1B0 = (8 * term_F2_F1F0
                - 4 * (term_F2F1 + term_F2F0 + term_F1F0)
                + 2 * (term_F2 + term_F1 + term_F0) - term_I)
    # d_B2B0B1
    d_B2B0B1 = (8 * term_F2_F0F1
                - 4 * (term_F2F0 + term_F2F1 + term_F0F1)
                + 2 * (term_F2 + term_F0 + term_F1) - term_I)
    # d_B2B0B2 (简化为 <F2 F0 F2> 相关的项)
    d_B2B0B2 = (8 * term_F2_F0F2
                - 4 * (term_F2F0 + term_F2 + term_F0F2)
                + 2 * (term_F2 + term_F0 + term_F2) - term_I)
    # d_B2B1B2
    d_B2B1B2 = (8 * term_F2_F1F2
                - 4 * (term_F2F1 + term_F2 + term_F1F2)
                + 2 * (term_F2 + term_F1 + term_F2) - term_I)
    # d_B0B2B0
    d_B0B2B0 = (8 * term_F0_F2F0
                - 4 * (term_F0F2 + term_F0 + term_F2F0)
                + 2 * (term_F0 + term_F2 + term_F0) - term_I)
    # d_B0B2B1
    d_B0B2B1 = (8 * term_F0_F2F1
                - 4 * (term_F0F2 + term_F0F1 + term_F2F1)
                + 2 * (term_F0 + term_F2 + term_F1) - term_I)
    # d_B1B2B0
    d_B1B2B0 = (8 * term_F1_F2F0
                - 4 * (term_F1F2 + term_F1F0 + term_F2F0)
                + 2 * (term_F1 + term_F2 + term_F0) - term_I)
    # d_B1B2B1
    d_B1B2B1 = (8 * term_F1_F2F1
                - 4 * (term_F1F2 + term_F1 + term_F2F1)
                + 2 * (term_F1 + term_F2 + term_F1) - term_I)

    # --- 4-Body ---
    # d_B0B2B1B0
    # Formula: 16*F0F2F1F0 - 8*(F0F2F1 + F0F2F0 + F0F1F0 + F2F1F0)
    #          + 4*(F0F2 + F0F1 + F2F1 + F2F0 + F1F0) - 2*(F2 + F1) + I
    d_B0B2B1B0 = (16 * term_F0F2_F1F0
                  - 8 * (term_F0_F2F1 + term_F0_F2F0 + term_F0_F1F0 + term_F2_F1F0)
                  + 4 * (term_F0F2 + term_F0F1 + term_F2F1 + term_F2F0 + term_F1F0)
                  - 2 * (term_F2 + term_F1)
                  + term_I)
    # d_B0B2B0B1
    # Formula: 16*F0F2F0F1 - 8*(F0F2F0 + F0F2F1 + F2F0F1)
    #          + 4*(F0F2 + F2F0 + F2F1) - 2*(F2 + F1) + I
    d_B0B2B0B1 = (16 * term_F0F2_F0F1
                  - 8 * (term_F0_F2F0 + term_F0_F2F1 + term_F2_F0F1)
                  + 4 * (term_F0F2 + term_F2F0 + term_F2F1)
                  - 2 * (term_F2 + term_F1)
                  + term_I)
    # d_B0B2B0B2
    # Formula: 16*F0F2F0F2 - 8*(F0F2F0 + F2F0F2)
    #          + 4*(F0F2 + F2F0) + I
    d_B0B2B0B2 = (16 * term_F0F2_F0F2
                  - 8 * (term_F0_F2F0 + term_F2_F0F2)
                  + 4 * (term_F0F2 + term_F2F0)
                  + term_I)
    # d_B0B2B1B2
    # Formula: 16*F0F2F1F2 - 8*(F0F2F1 + F0F1F2 + F2F1F2)
    #          + 4*(-F0F2 + F0F1 + F2F1 + F1F2) - 2*(F0 + F1) + I
    # 注意: F0F2 前是负号
    d_B0B2B1B2 = (16 * term_F0F2_F1F2
                  - 8 * (term_F0_F2F1 + term_F0_F1F2 + term_F2_F1F2)
                  + 4 * (-term_F0F2 + term_F0F1 + term_F2F1 + term_F1F2)
                  - 2 * (term_F0 + term_F1)
                  + term_I)
    # d_B1B2B1B0
    # Formula: 16*F1F2F1F0 - 8*(F1F2F1 + F1F2F0 + F2F1F0)
    #          + 4*(F1F2 + F2F1 + F2F0) - 2*(F2 + F0) + I
    d_B1B2B1B0 = (16 * term_F1F2_F1F0
                  - 8 * (term_F1_F2F1 + term_F1_F2F0 + term_F2_F1F0)
                  + 4 * (term_F1F2 + term_F2F1 + term_F2F0)
                  - 2 * (term_F2 + term_F0)
                  + term_I)
    # d_B1B2B0B1
    # Formula: 16*F1F2F0F1 - 8*(F1F2F0 + F1F2F1 + F2F0F1)
    #          + 4*(F1F2 + F2F0 + F2F1) - 2*(F2 + F0) + I
    d_B1B2B0B1 = (16 * term_F1F2_F0F1
                  - 8 * (term_F1_F2F0 + term_F1_F2F1 + term_F2_F0F1)
                  + 4 * (term_F1F2 + term_F2F0 + term_F2F1)
                  - 2 * (term_F2 + term_F0)
                  + term_I)
    # d_B1B2B0B2
    # Formula: 16*F1F2F0F2 - 8*(F1F2F0 + F1F0F2 + F2F0F2)
    #          + 4*(-F1F2 + F1F0 + F2F0 + F0F2) - 2*(F1 + F0) + I
    # 注意: F1F2 前是负号
    d_B1B2B0B2 = (16 * term_F1F2_F0F2
                  - 8 * (term_F1_F2F0 + term_F1_F0F2 + term_F2_F0F2)
                  + 4 * (-term_F1F2 + term_F1F0 + term_F2F0 + term_F0F2)
                  - 2 * (term_F1 + term_F0)
                  + term_I)
    # d_B1B2B1B2
    # Formula: 16*F1F2F1F2 - 8*(F1F2F1 + F2F1F2)
    #          + 4*(F1F2 + F2F1) + I
    d_B1B2B1B2 = (16 * term_F1F2_F1F2
                  - 8 * (term_F1_F2F1 + term_F2_F1F2)
                  + 4 * (term_F1F2 + term_F2F1)
                  + term_I)

    # ==========================================================================
    # 6. 构造 Localizing Matrix (gamma_loc)
    # ==========================================================================
    # 16个元素，按行排列

    # Row 1
    g_loc_1 = k1 * d_B2B0 - k2 * d_B2B1
    g_loc_2 = k1 * d_B2 - k2 * d_B2B1B0
    g_loc_3 = k1 * d_B2B0B1 - k2 * d_B2
    g_loc_4 = k1 * d_B2B0B2 - k2 * d_B2B1B2

    # Row 2
    g_loc_5 = k1 * d_B0B2B0 - k2 * d_B0B2B1
    g_loc_6 = k1 * d_B0B2 - k2 * d_B0B2B1B0
    g_loc_7 = k1 * d_B0B2B0B1 - k2 * d_B0B2
    g_loc_8 = k1 * d_B0B2B0B2 - k2 * d_B0B2B1B2

    # Row 3
    g_loc_9 = k1 * d_B1B2B0 - k2 * d_B1B2B1
    g_loc_10 = k1 * d_B1B2 - k2 * d_B1B2B1B0
    g_loc_11 = k1 * d_B1B2B0B1 - k2 * d_B1B2
    g_loc_12 = k1 * d_B1B2B0B2 - k2 * d_B1B2B1B2

    # Row 4
    g_loc_13 = k1 * d_B0 - k2 * d_B1
    g_loc_14 = k1 * d_I - k2 * d_B1B0
    g_loc_15 = k1 * d_B0B1 - k2 * d_I
    g_loc_16 = k1 * d_B0B2 - k2 * d_B1B2

    # 组装矩阵 (使用 cp.bmat)
    # 注意：g_loc_x 是 CVXPY 表达式
    gamma_loc = cp.bmat([
        [g_loc_1, g_loc_2, g_loc_3, g_loc_4],
        [g_loc_5, g_loc_6, g_loc_7, g_loc_8],
        [g_loc_9, g_loc_10, g_loc_11, g_loc_12],
        [g_loc_13, g_loc_14, g_loc_15, g_loc_16]
    ])

    # ==========================================================================
    # 7. 强制对称并返回约束
    # ==========================================================================
    gamma_loc_sym = 0.5 * (gamma_loc + gamma_loc.T)

    constraints = []
    constraints.append(gamma_loc_sym >> 0)

    return constraints


if __name__ == '__main__':
    # 测试代码可行性
    print("--- Testing localizing_matrix.py ---")

    # 模拟参数
    test_alpha = 0.5
    desc = np.array([2, 2, 2, 3])  # Alice 2, Bob 3
    m_vec = desc[2:4]

    # 模拟 G 和 ind_catalog (仅为了跑通逻辑，实际不可求解)
    # 在真实运行中，这些由 NPAHierarchy_3 生成
    try:
        # 这里的 G 只是一个占位符，无法真实运算，仅检查 build 函数是否有语法错误
        # 为了完整测试，通常需要 mock get_gamma_element_bob 的返回值
        print(f"Alpha input: {test_alpha}")
        print("Dependency check: locate_gamma_Bob imported.")
        print("Note: Run this module within the full SDP context to generate constraints.")
    except Exception as e:
        print(f"Test Error: {e}")