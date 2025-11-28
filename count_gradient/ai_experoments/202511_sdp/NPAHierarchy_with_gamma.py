import numpy as np
import cvxpy as cp
import sys
from typing import Union, List, Tuple, Optional

# 假设辅助函数保存在 npa_support.py 中
from npa_support import find_in_cell, update_ind, product_of_orthogonal


def npa_hierarchy(cg: np.ndarray, desc: Union[List[int], np.ndarray], k: Union[int, str] = 1) -> Tuple[
    float, Optional[np.ndarray], List[np.ndarray]]:
    """
    执行 NPA Hierarchy 检查，并返回结果、Gamma 矩阵和索引目录。

    返回:
        (is_npa, G_value, ind_catalog)
        - is_npa: 1.0 (Pass) or 0.0 (Fail)
        - G_value: 求解后的 Gamma 矩阵 (numpy array)，如果失败则为 None
        - ind_catalog: 算符序列目录，用于解读 Gamma 矩阵的行列
    """

    # --- 1. 参数 K 解析 ---
    base_k = 1
    num_k_compon = 1
    k_types = []

    if isinstance(k, (int, float)):
        base_k = int(k)
        num_k_compon = 1
        level_val = base_k
    elif isinstance(k, str):
        k_types = k.lower().replace(' ', '').split('+')
        try:
            base_k = int(k_types[0])
        except ValueError:
            raise ValueError("NPAHierarchy: K 字符串的第一个部分必须是数字。")

        num_k_compon = len(k_types)
        if num_k_compon > 1:
            level_val = base_k
            for j in range(1, num_k_compon):
                level_val = max(level_val, len(k_types[j]))
        else:
            level_val = base_k
    else:
        raise ValueError("NPAHierarchy: K 必须是正整数或字符串。")

    # --- 2. 场景参数解析 ---
    desc = np.array(desc, dtype=int).flatten()
    o_vec = desc[0:2]
    m_vec = desc[2:4]

    aindex = lambda a, x: 1 + a + x * (o_vec[0] - 1)
    bindex = lambda b, y: 1 + b + y * (o_vec[1] - 1)

    o_vec_minus_1 = o_vec - 1
    tot_dim = 1 + int(np.dot(o_vec_minus_1, m_vec)) ** level_val

    tol = np.finfo(float).eps ** (0.75)

    # --- 3. 生成 NPA 矩阵的索引目录 ---
    i_ind = np.vstack([np.zeros(level_val, dtype=int), -np.ones(level_val, dtype=int)])
    j_ind = i_ind.copy()

    ind_catalog: List[np.ndarray] = []
    o_vec_limit = o_vec - 2

    for j in range(tot_dim):
        res, res_type = product_of_orthogonal(j_ind, m_vec)
        res_fnd = find_in_cell(res, ind_catalog)

        if res_fnd == -1 and res_type != 0:
            is_valid_res = (res.shape[1] <= base_k)
            if not is_valid_res and num_k_compon >= 2:
                num_a_res = np.sum(res[0, :] < m_vec[0])
                num_b_res = res.shape[1] - num_a_res
                for i in range(1, num_k_compon):
                    k_str = k_types[i]
                    num_a_fnd = k_str.count('a')
                    num_b_fnd = k_str.count('b')
                    if num_a_res <= num_a_fnd and num_b_res <= num_b_fnd:
                        is_valid_res = True
                        break
            if is_valid_res:
                ind_catalog.append(res)

        j_ind = update_ind(j_ind, level_val, m_vec, o_vec_limit)

    real_dim = len(ind_catalog)

    # --- 4. 构建并求解 SDP ---
    G = cp.Variable((real_dim, real_dim), symmetric=True)
    constraints = [G >> 0]

    res_catalog: List[np.ndarray] = []
    res_loc: List[List[int]] = []

    for i in range(real_dim):
        for j in range(i, real_dim):
            term_i = ind_catalog[i][:, ::-1]
            term_j = ind_catalog[j]
            combined_ind = np.hstack([term_i, term_j])

            res, res_type = product_of_orthogonal(combined_ind, m_vec)

            if res_type == 0:
                constraints.append(G[i, j] == 0)
            elif res_type == 2:
                idx_a = aindex(res[1, 0], res[0, 0])
                idx_b = bindex(res[1, 1], res[0, 1] - m_vec[0])
                constraints.append(G[i, j] == cg[idx_a, idx_b])
            elif res_type == 1:
                if np.array_equal(res, np.array([[0], [-1]])):
                    constraints.append(G[i, j] == 1)
                elif res[0, 0] >= m_vec[0]:
                    idx_b_marg = bindex(res[1, 0], res[0, 0] - m_vec[0])
                    constraints.append(G[i, j] == cg[0, idx_b_marg])
                else:
                    idx_a_marg = aindex(res[1, 0], res[0, 0])
                    constraints.append(G[i, j] == cg[idx_a_marg, 0])
            else:
                res_fnd = find_in_cell(res, res_catalog)
                if res_fnd == -1:
                    res_conj, _ = product_of_orthogonal(res[:, ::-1], m_vec)
                    res_fnd = find_in_cell(res_conj, res_catalog)
                if res_fnd == -1:
                    res_catalog.append(res)
                    res_loc.append([i, j])
                else:
                    prev_loc = res_loc[res_fnd]
                    constraints.append(G[i, j] == G[prev_loc[0], prev_loc[1]])

    # --- 5. 求解问题 ---
    prob = cp.Problem(cp.Minimize(0), constraints)

    try:
        prob.solve(solver=cp.SCS, eps=tol)
    except cp.SolverError:
        return 0.0, None, ind_catalog

    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        # 修改返回值：返回 is_npa, G矩阵数值, 索引目录
        return 1.0, G.value, ind_catalog
    else:
        return 0.0, None, ind_catalog