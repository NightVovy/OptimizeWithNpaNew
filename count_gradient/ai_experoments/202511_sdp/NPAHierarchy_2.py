import numpy as np
import cvxpy as cp
from typing import Union, List, Tuple, Optional
from npa_support import find_in_cell, update_ind, product_of_orthogonal


def npa_constraints(cg: np.ndarray, desc: Union[List[int], np.ndarray], k: Union[int, str] = 1):
    """
    【核心构建函数】
    只构建变量、约束和目录，不进行求解。
    供其他需要自定义目标函数的 SDP 调用（如 fidelity_chsh）。

    返回:
        (G_var, constraints, ind_catalog, m_vec)
        - G_var: CVXPY 变量对象 (Gamma矩阵)
        - constraints: 约束列表 (包含 NPA约束 和 数据一致性约束)
        - ind_catalog: 索引目录
        - m_vec: 测量设置数 (方便后续查找用)
    """
    # --- 1. 参数解析 (同原代码) ---
    # ... (K解析代码省略，与之前相同，为节省篇幅) ...
    # 简化的 K 解析（假设输入是整数）
    base_k = int(k) if isinstance(k, (int, float)) else 1
    level_val = base_k
    num_k_compon = 1
    k_types = []  # 暂略复杂字符串解析

    desc = np.array(desc, dtype=int).flatten()
    o_vec = desc[0:2]
    m_vec = desc[2:4]

    aindex = lambda a, x: 1 + a + x * (o_vec[0] - 1)
    bindex = lambda b, y: 1 + b + y * (o_vec[1] - 1)

    o_vec_minus_1 = o_vec - 1
    tot_dim = 1 + int(np.dot(o_vec_minus_1, m_vec)) ** level_val

    # --- 2. 生成目录 ---
    i_ind = np.vstack([np.zeros(level_val, dtype=int), -np.ones(level_val, dtype=int)])
    j_ind = i_ind.copy()
    ind_catalog: List[np.ndarray] = []
    o_vec_limit = o_vec - 2

    for j in range(tot_dim):
        res, res_type = product_of_orthogonal(j_ind, m_vec)
        res_fnd = find_in_cell(res, ind_catalog)
        if res_fnd == -1 and res_type != 0:
            if res.shape[1] <= base_k:  # 简化判断
                ind_catalog.append(res)
        j_ind = update_ind(j_ind, level_val, m_vec, o_vec_limit)

    real_dim = len(ind_catalog)

    # --- 3. 构建变量和约束 ---
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
            elif res_type == 2:  # 数据一致性约束 (即你的第二个subject to)
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

    # 返回所有构建块，但不求解
    return G, constraints, ind_catalog, m_vec


# 原有的函数现在变成了一个简单的包装器
def npa_hierarchy(cg: np.ndarray, desc: Union[List[int], np.ndarray], k: Union[int, str] = 1):
    # 1. 调用构建器拿到变量和约束
    G, constraints, ind_catalog, _ = npa_constraints(cg, desc, k)

    # 2. 求解 (仅判断可行性)
    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver=cp.SCS, eps=1e-5)
    except cp.SolverError:
        return 0.0, None, ind_catalog

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return 1.0, G.value, ind_catalog
    else:
        return 0.0, None, ind_catalog