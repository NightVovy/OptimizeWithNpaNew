import numpy as np
import cvxpy as cp
import sys
from typing import Union, List, Tuple, Optional

try:
    from npa_support import find_in_cell, update_ind, product_of_orthogonal
except ImportError:
    print("错误: 无法导入 npa_support")
    sys.exit(1)


def npa_constraints(cg: np.ndarray, desc: Union[List[int], np.ndarray], k: Union[int, str] = 1,
                    enforce_data: bool = True,
                    custom_basis: Optional[List[Tuple[int, ...]]] = None) -> Tuple[
    cp.Variable, List[cp.Constraint], List[np.ndarray]]:
    """
    【构建器 V6 (自定义基底版)】

    新增参数:
        custom_basis: 一个列表，包含允许的算符序列(元组形式)。
                      如果提供了此参数，k 参数将被忽略(仅用于确定最大长度)。
                      例如: [(0,), (1,), (0,1), (2,)] 代表 E0, E1, E0E1, F0
    """

    # ==========================================================================
    # 1. 参数解析
    # ==========================================================================
    # 如果有自定义基底，我们需要先确定最大长度 k，以便生成循环能跑够次数
    if custom_basis is not None:
        # 自动计算最大长度
        max_len = 0
        for seq in custom_basis:
            if seq:  # 忽略空元组
                max_len = max(max_len, len(seq))
        level_val = max(1, max_len)
        base_k = 0  # 在自定义模式下，base_k 设为0，完全依赖白名单

        # 将 list 转为 set 以便 O(1) 查找
        allowed_seqs_set = set(custom_basis)
        print(f"[NPA] 使用自定义基底模式。最大长度: {level_val}, 基底数: {len(allowed_seqs_set)}")

    else:
        # --- 原有的 K 解析逻辑 (省略以节省篇幅，保持 V5 版本逻辑) ---
        # (此处保留 V5 的 k='1+aaa' 解析代码，为了简洁我略写了，实际使用请保留)
        if isinstance(k, str):
            parts = k.lower().split('+')
            base_k = int(parts[0].strip())
            level_val = base_k
            for p in parts[1:]:
                level_val = max(level_val, len(p.strip()))
        else:
            base_k = int(k)
            level_val = base_k

    # ==========================================================================
    # 2. 场景参数解析
    # ==========================================================================
    desc = np.array(desc, dtype=int).flatten()
    o_vec = desc[0:2]
    m_vec = desc[2:4]

    # ... (aindex, bindex 定义不变) ...
    aindex = lambda a, x: 1 + a + x * (o_vec[0] - 1)
    bindex = lambda b, y: 1 + b + y * (o_vec[1] - 1)

    o_vec_minus_1 = o_vec - 1
    tot_dim = 1 + int(np.dot(o_vec_minus_1, m_vec)) ** level_val

    # ==========================================================================
    # 3. 生成目录 (核心修改: 增加白名单筛选)
    # ==========================================================================
    i_ind = np.vstack([np.zeros(level_val, dtype=int), -np.ones(level_val, dtype=int)])
    j_ind = i_ind.copy()
    ind_catalog: List[np.ndarray] = []
    o_vec_limit = o_vec - 2

    for j in range(tot_dim):
        res, res_type = product_of_orthogonal(j_ind, m_vec)
        res_fnd = find_in_cell(res, ind_catalog)

        if res_fnd == -1 and res_type != 0:

            # --- 筛选逻辑 ---
            is_valid = False

            # 提取当前算符的序列索引 (Row 0)
            # 例如 E0 -> [0], F0 -> [2]
            current_seq = tuple(res[0, :])

            # [分支 A] 自定义基底模式
            if custom_basis is not None:
                # 恒等算符 (空序列) 永远保留
                if len(current_seq) == 0:
                    is_valid = True
                # 检查是否在白名单中
                elif current_seq in allowed_seqs_set:
                    is_valid = True

            # [分支 B] 常规 K 模式
            else:
                # ... (V5 的逻辑: 检查长度 <= base_k 或匹配 pattern) ...
                if res.shape[1] <= base_k:
                    is_valid = True
                # (这里省略了 pattern 代码，实际请保留 V5 的内容)

            if is_valid:
                ind_catalog.append(res)

        j_ind = update_ind(j_ind, level_val, m_vec, o_vec_limit)

    real_dim = len(ind_catalog)
    # print(f"生成 Gamma 矩阵维度: {real_dim}x{real_dim}")

    # ==========================================================================
    # 4. 构建变量和约束 (保持 Robustness 强约束)
    # ==========================================================================
    G = cp.Variable((real_dim, real_dim), symmetric=True)
    constraints = []

    # 1. 半正定
    constraints.append(G >> 0)
    # 2. 归一化 (强制 I=1)
    constraints.append(G[0, 0] == 1)
    # 下面的已经注释掉好几次了...
    # 3. 对角线约束
    constraints.append(cp.diag(G) <= 1)
    constraints.append(cp.diag(G) >= 0)
    # 4. 全局数值笼子
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

            if res_type == 0:
                constraints.append(G[i, j] == 0)
                continue
            if np.array_equal(res, np.array([[0], [-1]])):
                constraints.append(G[i, j] == 1)
                continue

            res_fnd = find_in_cell(res, res_catalog)
            if res_fnd == -1 and res_type == -1:
                res_conj, _ = product_of_orthogonal(res[:, ::-1], m_vec)
                res_fnd = find_in_cell(res_conj, res_catalog)

            if res_fnd != -1:
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