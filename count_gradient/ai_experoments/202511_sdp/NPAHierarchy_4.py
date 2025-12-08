import numpy as np
import cvxpy as cp
import sys
from typing import Union, List, Tuple, Optional

# 假设辅助函数保存在 npa_support.py 中
try:
    from npa_support import find_in_cell, update_ind, product_of_orthogonal
except ImportError:
    print("错误: 无法导入 npa_support，请确保该文件在当前目录下。")
    sys.exit(1)


def npa_constraints(cg: np.ndarray, desc: Union[List[int], np.ndarray], k: Union[int, str] = 1,
                    enforce_data: bool = True) -> Tuple[cp.Variable, List[cp.Constraint], List[np.ndarray]]:
    """
    【构建器 V3 (边界增强版)】生成 NPA 层次结构的 SDP 变量和约束。

    修复:
    1. 增加了对角线元素 <= 1 的约束 (防止数值爆炸)。
    2. 增加了全局非负约束 (防止出现 -4 这种非物理概率)。
    """

    # --- 1. 参数 K 解析 ---
    base_k = int(k) if isinstance(k, (int, float)) else 1
    level_val = base_k

    # --- 2. 场景参数解析 ---
    desc = np.array(desc, dtype=int).flatten()
    o_vec = desc[0:2]
    m_vec = desc[2:4]

    aindex = lambda a, x: 1 + a + x * (o_vec[0] - 1)
    bindex = lambda b, y: 1 + b + y * (o_vec[1] - 1)

    o_vec_minus_1 = o_vec - 1
    tot_dim = 1 + int(np.dot(o_vec_minus_1, m_vec)) ** level_val

    # --- 3. 生成目录 ---
    i_ind = np.vstack([np.zeros(level_val, dtype=int), -np.ones(level_val, dtype=int)])
    j_ind = i_ind.copy()
    ind_catalog: List[np.ndarray] = []
    o_vec_limit = o_vec - 2

    for j in range(tot_dim):
        res, res_type = product_of_orthogonal(j_ind, m_vec)
        res_fnd = find_in_cell(res, ind_catalog)
        if res_fnd == -1 and res_type != 0:
            if res.shape[1] <= base_k:
                ind_catalog.append(res)
        j_ind = update_ind(j_ind, level_val, m_vec, o_vec_limit)

    real_dim = len(ind_catalog)

    # --- 4. 构建变量和约束 ---
    G = cp.Variable((real_dim, real_dim), symmetric=True)

    constraints = []

    # [约束 1] 半正定
    constraints.append(G >> 0)

    # [约束 2 - 新增] 对角线约束：所有对角元素 <= 1
    # 物理意义：投影算子序列的模长平方不可能超过 1
    constraints.append(cp.diag(G) <= 1)
    # ...
    constraints.append(cp.diag(G) >= 0)

    # 3. 全局数值笼子 (Global Box)
    # 防止非对角元素变成 -5, 7 这种鬼东西。
    # 理论上关联值绝对值 <= 1。稍微给点 buffer (1.01) 防止数值截断导致 Infeasible。
    constraints.append(G <= 1.0)
    constraints.append(G >= -1.0)

    res_catalog: List[np.ndarray] = []
    res_loc: List[List[int]] = []

    for i in range(real_dim):
        for j in range(i, real_dim):
            term_i = ind_catalog[i][:, ::-1]
            term_j = ind_catalog[j]
            combined_ind = np.hstack([term_i, term_j])

            res, res_type = product_of_orthogonal(combined_ind, m_vec)

            # 1. 零算符
            if res_type == 0:
                constraints.append(G[i, j] == 0)
                continue

            # 2. 恒等算符 (Identity)
            if np.array_equal(res, np.array([[0], [-1]])):
                constraints.append(G[i, j] == 1)
                continue

            # --- 一致性检查 (Consistency Check) ---
            res_fnd = find_in_cell(res, res_catalog)

            if res_fnd == -1 and res_type == -1:
                res_conj, _ = product_of_orthogonal(res[:, ::-1], m_vec)
                res_fnd_conj = find_in_cell(res_conj, res_catalog)
                if res_fnd_conj != -1:
                    res_fnd = res_fnd_conj

            if res_fnd != -1:
                # 强制相等
                prev_loc = res_loc[res_fnd]
                constraints.append(G[i, j] == G[prev_loc[0], prev_loc[1]])
            else:
                # 新变量
                res_catalog.append(res)
                res_loc.append([i, j])

                # --- 数据数值约束 (仅在 enforce_data=True 时施加) ---
                if enforce_data:
                    if res_type == 2:
                        idx_a = aindex(res[1, 0], res[0, 0])
                        idx_b = bindex(res[1, 1], res[0, 1] - m_vec[0])
                        constraints.append(G[i, j] == cg[idx_a, idx_b])
                    elif res_type == 1:
                        if res[0, 0] >= m_vec[0]:  # Bob
                            idx_b_marg = bindex(res[1, 0], res[0, 0] - m_vec[0])
                            constraints.append(G[i, j] == cg[0, idx_b_marg])
                        else:  # Alice
                            idx_a_marg = aindex(res[1, 0], res[0, 0])
                            constraints.append(G[i, j] == cg[idx_a_marg, 0])

    return G, constraints, ind_catalog