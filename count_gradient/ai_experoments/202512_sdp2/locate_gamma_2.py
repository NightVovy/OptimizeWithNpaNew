import numpy as np
import cvxpy as cp  # 引入 cvxpy 以支持类型检查
from typing import Union, List, Any, Optional
from npa_support import find_in_cell, product_of_orthogonal


def get_gamma_element(G: Union[np.ndarray, cp.Variable],
                      ind_catalog: list,
                      m_vec: np.ndarray,
                      alice_seq: list,
                      bob_seq: list) -> Union[float, cp.Expression, None]:
    """
    在 Gamma 矩阵/变量中查找 <Alice_Seq * Bob_Seq> 对应的元素。

    通用化版本：
    1. 如果 G 是 numpy 数组，返回 float 数值。
    2. 如果 G 是 cvxpy 变量，返回 cvxpy 表达式 (用于构建 SDP)。

    参数:
        G: Gamma 矩阵 (数值或变量)
        ind_catalog: 算符目录
        m_vec: 测量设置向量 [MA, MB]
        alice_seq: Alice 的投影算子设置索引列表 (例如 [0, 1, 0])
        bob_seq: Bob 的投影算子设置索引列表 (注意：需要使用全局索引，例如 Alice有2个设置，Bob设置0为索引2)

    返回:
        对应的矩阵元素 G[u, v]。如果找不到，返回 None。
    """

    # 1. 构造 Alice 的查找目标 (对应 Gamma 的行 u)
    # S_u = (Alice_Seq)^dagger = Reverse(Alice_Seq)
    alice_input_rev = alice_seq[::-1]

    if len(alice_input_rev) > 0:
        res_a_raw = np.vstack([alice_input_rev, np.zeros(len(alice_input_rev), dtype=int)])
    else:
        # 空列表代表恒等算符
        res_a_raw = np.array([[0], [-1]])

    res_a, _ = product_of_orthogonal(res_a_raw, m_vec)
    row_idx = find_in_cell(res_a, ind_catalog)

    # 2. 构造 Bob 的查找目标 (对应 Gamma 的列 v)
    if len(bob_seq) > 0:
        res_b_raw = np.vstack([bob_seq, np.zeros(len(bob_seq), dtype=int)])
    else:
        res_b_raw = np.array([[0], [-1]])

    res_b, _ = product_of_orthogonal(res_b_raw, m_vec)
    col_idx = find_in_cell(res_b, ind_catalog)

    # 3. 返回结果
    if row_idx != -1 and col_idx != -1:
        # 直接返回切片，无论是数值还是表达式
        return G[row_idx, col_idx]
    else:
        # 在构建阶段打印错误有助于调试
        print(f"[Warning] 算符序列未在 ind_catalog 中找到 (K可能不足):")
        if row_idx == -1: print(f"  Alice (Simplified): {res_a.flatten()}")
        if col_idx == -1: print(f"  Bob (Simplified):   {res_b.flatten()}")
        return None