import numpy as np
import cvxpy as cp
from typing import Union, List, Any, Optional
from npa_support import find_in_cell, product_of_orthogonal


def get_gamma_element_bob(G: Union[np.ndarray, cp.Variable],
                          ind_catalog: list,
                          m_vec: np.ndarray,
                          bob_seq_row: list,
                          bob_seq_col: list) -> Union[float, cp.Expression, None]:
    """
    在 Gamma 矩阵/变量中查找纯 Bob 算符组合 <Bob_Seq_Row * Bob_Seq_Col> 对应的元素。

    原理: Gamma[u, v] = <S_u^dagger * S_v>
    我们令:
      Target = < Bob_Seq_Row * Bob_Seq_Col >
      S_v = Bob_Seq_Col
      S_u^dagger = Bob_Seq_Row
      => S_u = (Bob_Seq_Row)^dagger = Reverse(Bob_Seq_Row) (因为投影算子是厄米的)

    参数:
        G: Gamma 矩阵 (数值或变量)
        ind_catalog: 算符目录
        m_vec: 测量设置向量 [MA, MB] (即使只查Bob，也需要这个来确定全局索引的分界线)
        bob_seq_row: 作为左乘项的 Bob 投影算子设置索引列表 (全局索引)
        bob_seq_col: 作为右乘项的 Bob 投影算子设置索引列表 (全局索引)

        [重要提示]：输入的索引必须是全局索引。
        例如 desc=[2,2,2,2] (Alice 2 inputs, Bob 2 inputs):
        - Alice 索引: 0, 1
        - Bob 索引: 2, 3
        如果你想查 Bob 的第 0 个设置，这里必须输入 [2]。

    返回:
        对应的矩阵元素 G[u, v]。如果找不到，返回 None。
    """

    # ==========================================================================
    # 1. 构造行的查找目标 (对应 Gamma 的行 u)
    # ==========================================================================
    # S_u = (Row_Seq)^dagger。我们需要反转输入序列。
    bob_row_rev = bob_seq_row[::-1]

    if len(bob_row_rev) > 0:
        # Row 0: 设置索引, Row 1: 结果索引(默认为0)
        res_row_raw = np.vstack([bob_row_rev, np.zeros(len(bob_row_rev), dtype=int)])
    else:
        # 空列表代表恒等算符
        res_row_raw = np.array([[0], [-1]])

    # 简化算符
    # product_of_orthogonal 会根据 m_vec 判断这些索引属于 Bob，并进行相应的代数简化
    res_row, _ = product_of_orthogonal(res_row_raw, m_vec)
    row_idx = find_in_cell(res_row, ind_catalog)

    # ==========================================================================
    # 2. 构造列的查找目标 (对应 Gamma 的列 v)
    # ==========================================================================
    # v = Col_Seq (不需要反转)
    if len(bob_seq_col) > 0:
        res_col_raw = np.vstack([bob_seq_col, np.zeros(len(bob_seq_col), dtype=int)])
    else:
        res_col_raw = np.array([[0], [-1]])

    res_col, _ = product_of_orthogonal(res_col_raw, m_vec)
    col_idx = find_in_cell(res_col, ind_catalog)

    # ==========================================================================
    # 3. 返回结果
    # ==========================================================================
    if row_idx != -1 and col_idx != -1:
        return G[row_idx, col_idx]
    else:
        # 调试信息 (可选开启)
        print(f"[Warning-Bob] 算符序列未在 ind_catalog 中找到 (K可能不足):")
        if row_idx == -1: print(f"  Row (Simplified): {res_row.flatten()}")
        if col_idx == -1: print(f"  Col (Simplified): {res_col.flatten()}")
        return None