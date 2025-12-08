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
    【构建器 V5 (字符串支持 + 边界增强版)】

    新增功能:
    1. 支持 k 输入为字符串 (例如 '1+aaa+bbb')，用于生成混合层级。

    保留功能:
    1. 强制对角线元素 <= 1。
    2. 全局数值笼子 [-1, 1]。
    """

    # ==========================================================================
    # 1. 参数 K 解析 (新增字符串处理逻辑)
    # ==========================================================================
    k_patterns = []  # 用于存储 (num_alice, num_bob) 的允许模式

    if isinstance(k, str):
        # 解析字符串，例如 '1+ab+aab'
        parts = k.lower().split('+')

        # 第一部分必须是基础整数层级 (Base Level)
        try:
            base_k = int(parts[0].strip())
        except ValueError:
            raise ValueError("K 字符串的第一部分必须是整数 (Base Level)。")

        level_val = base_k  # level_val 代表生成目录时所需的最大长度

        # 解析后续模式
        for p in parts[1:]:
            p = p.strip()
            if not p: continue

            num_a = p.count('a')
            num_b = p.count('b')
            k_patterns.append((num_a, num_b))

            # 更新最大长度，以确保循环能覆盖到这些长算符
            level_val = max(level_val, len(p))

    else:
        # 普通整数模式
        base_k = int(k)
        level_val = base_k

    # ==========================================================================
    # 2. 场景参数解析
    # ==========================================================================
    desc = np.array(desc, dtype=int).flatten()
    o_vec = desc[0:2]
    m_vec = desc[2:4]

    aindex = lambda a, x: 1 + a + x * (o_vec[0] - 1)
    bindex = lambda b, y: 1 + b + y * (o_vec[1] - 1)

    o_vec_minus_1 = o_vec - 1

    # 使用 level_val (最大长度) 计算总维度上限
    tot_dim = 1 + int(np.dot(o_vec_minus_1, m_vec)) ** level_val

    # ==========================================================================
    # 3. 生成目录 (增加了模式筛选逻辑)
    # ==========================================================================
    i_ind = np.vstack([np.zeros(level_val, dtype=int), -np.ones(level_val, dtype=int)])
    j_ind = i_ind.copy()
    ind_catalog: List[np.ndarray] = []
    o_vec_limit = o_vec - 2

    for j in range(tot_dim):
        res, res_type = product_of_orthogonal(j_ind, m_vec)
        res_fnd = find_in_cell(res, ind_catalog)

        # 如果是一个新算符且非零
        if res_fnd == -1 and res_type != 0:

            # --- [新增] 筛选逻辑 ---
            current_len = res.shape[1]
            is_valid = False

            # A. 检查是否在基础层级内
            if current_len <= base_k:
                is_valid = True

            # B. 如果超出了基础层级，检查是否符合特定模式 (仅当有模式时)
            elif k_patterns:
                # 统计当前算符中 Alice 和 Bob 的测量个数
                # m_vec[0] 是 Alice 的测量数。小于它的索引属于 Alice，大于等于它的属于 Bob。
                num_alice_ops = np.sum(res[0, :] < m_vec[0])
                num_bob_ops = current_len - num_alice_ops

                # 检查是否匹配任意一个允许模式
                for (pat_a, pat_b) in k_patterns:
                    if num_alice_ops <= pat_a and num_bob_ops <= pat_b:
                        is_valid = True
                        break

            if is_valid:
                ind_catalog.append(res)

        j_ind = update_ind(j_ind, level_val, m_vec, o_vec_limit)

    real_dim = len(ind_catalog)

    # ==========================================================================
    # 4. 构建变量和约束 (保持之前的 Robustness 修复)
    # ==========================================================================
    G = cp.Variable((real_dim, real_dim), symmetric=True)

    constraints = []

    # [约束 1] 半正定
    constraints.append(G >> 0)

    # [约束 2] 对角线约束 (物理模长)
    constraints.append(cp.diag(G) <= 1)
    constraints.append(cp.diag(G) >= 0)

    # [约束 3] 全局数值笼子 (防止数值爆炸)
    constraints.append(G <= 1.0)
    constraints.append(G >= -1.0)

    # [约束 4] 归一化 (防止整体缩放)
    # 确保 ind_catalog[0] 是恒等算符 (通常 index -1,-1 -> res [[0],[-1]])
    # 这一步通常由 update_ind 保证 I 是第一个，但显式加上更安全
    # if len(ind_catalog) > 0 and np.array_equal(ind_catalog[0], np.array([[0], [-1]])):
    #    constraints.append(G[0, 0] == 1)

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

            # 2. 恒等算符
            if np.array_equal(res, np.array([[0], [-1]])):
                constraints.append(G[i, j] == 1)
                continue

            # --- 一致性检查 ---
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

                # --- 数据数值约束 ---
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