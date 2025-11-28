import numpy as np
import cvxpy as cp
import sys
from typing import Union, List, Optional


# 假设辅助函数保存在 npa_support.py 中
# 如果你在同一个文件中运行，请确保上面的辅助函数定义在此函数之前
from npa_support import find_in_cell, update_ind, product_of_orthogonal

def npa_hierarchy(cg: np.ndarray, desc: Union[List[int], np.ndarray], k: Union[int, str] = 1) -> float:
    """
    判断一组 Collins-Gisin (CG) 格式的相关性是否满足 NPA 层次结构的条件。

    参数:
        cg (np.ndarray): Collins-Gisin 格式的相关性矩阵。
        desc (list or np.ndarray): [OA, OB, MA, MB]。
            OA/OB: Alice/Bob 的结果数。
            MA/MB: Alice/Bob 的测量设置数。
        k (int or str, optional): NPA 层次的层级。默认为 1。

    返回:
        float: 1.0 表示满足 NPA 条件，0.0 表示不满足。
               (如果在 CVXPY 优化问题中使用，可能返回 Expression)
    """

    # --- 1. 参数 K 解析 ---
    # 解析输入参数 K 以确定使用哪些测量算符。
    base_k = 1
    num_k_compon = 1
    k_types = []

    if isinstance(k, (int, float)):
        base_k = int(k)
        num_k_compon = 1
        level_val = base_k
    elif isinstance(k, str):
        # Python 版本的解析字符串逻辑 (例如 '1+ab+aab')
        # 移除空格并按 '+' 分割
        k_types = k.lower().replace(' ', '').split('+')
        # 第一部分通常是基础层级数字
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
    # 确保 desc 是 numpy 数组
    desc = np.array(desc, dtype=int).flatten()
    o_vec = desc[0:2]  # [OA, OB]
    m_vec = desc[2:4]  # [MA, MB]

    # 定义索引映射函数 (Lambda)
    # 对应 MATLAB: 2 + a + x*(o_vec(1)-1)
    # Python (0-based): 1 + a + x*(o_vec[0]-1)
    # 注意: a, x 在物理意义上是从 0 开始的
    aindex = lambda a, x: 1 + a + x * (o_vec[0] - 1)
    bindex = lambda b, y: 1 + b + y * (o_vec[1] - 1)

    # 计算 tot_dim (上限)
    # MATLAB: 1 + ((o_vec-1)'*m_vec)^k
    # Python: dot product
    o_vec_minus_1 = o_vec - 1
    tot_dim = 1 + int(np.dot(o_vec_minus_1, m_vec)) ** level_val

    # 数值容差
    tol = np.finfo(float).eps ** (0.75)

    # --- 3. 生成 NPA 矩阵的索引目录 (Index Catalog) ---

    # 初始化迭代器 (2 x k)
    # MATLAB: [zeros(1,k); -ones(1,k)]
    # Python: 同样结构
    i_ind = np.vstack([np.zeros(level_val, dtype=int), -np.ones(level_val, dtype=int)])
    j_ind = i_ind.copy()

    ind_catalog: List[np.ndarray] = []

    # 循环生成所有需要的测量乘积
    # 注意: 我们只对【独立结果】生成算符。
    # 对于 OA 个结果，独立结果有 OA-1 个，其最大索引为 OA-2。
    # 例如 CHSH (OA=2): 独立结果数=1，最大索引=0。
    # update_ind 期待的是最大有效索引，所以这里传入 o_vec - 2
    o_vec_limit = o_vec - 2

    for j in range(tot_dim):
        # 简化算符乘积
        res, res_type = product_of_orthogonal(j_ind, m_vec)

        # 检查是否已存在于目录中
        # Python find_in_cell 返回 -1 表示未找到
        res_fnd = find_in_cell(res, ind_catalog)

        # 确保测量是 (1) 新的，且 (2) 根据用户输入 K 是有效的
        if res_fnd == -1 and res_type != 0:
            is_valid_res = (res.shape[1] <= base_k)

            # 如果 K 是复杂字符串，进行额外检查
            if not is_valid_res and num_k_compon >= 2:
                # 计算属于 Alice 的算符数量
                # res[0, :] 是测量设置索引。如果 < MA，则是 Alice。
                num_a_res = np.sum(res[0, :] < m_vec[0])
                num_b_res = res.shape[1] - num_a_res

                for i in range(1, num_k_compon):  # Python list index 1 to end
                    k_str = k_types[i]
                    num_a_fnd = k_str.count('a')
                    num_b_fnd = k_str.count('b')

                    if num_a_res <= num_a_fnd and num_b_res <= num_b_fnd:
                        is_valid_res = True
                        break

            if is_valid_res:
                ind_catalog.append(res)

        # 更新迭代器
        j_ind = update_ind(j_ind, level_val, m_vec, o_vec_limit)

    real_dim = len(ind_catalog)

    # --- 4. 构建并求解 SDP (使用 CVXPY) ---

    # 定义 SDP 变量 G (对称矩阵)
    G = cp.Variable((real_dim, real_dim), symmetric=True)

    constraints = []

    # 强制 G 为半正定
    constraints.append(G >> 0)

    # 临时存储非交换项的目录和位置
    res_catalog: List[np.ndarray] = []
    res_loc: List[List[int]] = []  # 存储 [row, col]

    # 双重循环遍历 G 的所有条目并施加约束
    for i in range(real_dim):
        for j in range(i, real_dim):  # 只遍历上三角，利用对称性

            # 确定当前矩阵元素对应的算符乘积类型
            # MATLAB: [fliplr(ind_catalog{i}), ind_catalog{j}]
            # Python: 拼接矩阵。ind_catalog[i] 需要左右翻转 (S_i^dagger)
            term_i = ind_catalog[i][:, ::-1]  # fliplr
            term_j = ind_catalog[j]

            combined_ind = np.hstack([term_i, term_j])

            res, res_type = product_of_orthogonal(combined_ind, m_vec)

            # --- 根据类型施加约束 ---

            # 1. 零算符 (Orthogonal)
            if res_type == 0:
                constraints.append(G[i, j] == 0)

            # 2. 双方关联 (Correlation <AB>)
            elif res_type == 2:
                # aindex/bindex 需要传入具体的 a, x, b, y
                # res 结构: [setting; result]
                # res 此时是 [Alice_col, Bob_col] (因为已被 sortrows)
                # Alice: col 0, Bob: col 1

                # a = res[1, 0], x = res[0, 0]
                idx_a = aindex(res[1, 0], res[0, 0])

                # b = res[1, 1], y = res[0, 1] - MA (因为 Bob 设置索引从 MA 开始)
                idx_b = bindex(res[1, 1], res[0, 1] - m_vec[0])

                constraints.append(G[i, j] == cg[idx_a, idx_b])

            # 3. 单体边缘概率 (Marginal <A> or <B> or Identity)
            elif res_type == 1:
                # 检查是否是恒等算符 [0; -1]
                if np.array_equal(res, np.array([[0], [-1]])):
                    constraints.append(G[i, j] == 1)  # Normalization <I> = 1

                elif res[0, 0] >= m_vec[0]:
                    # 测量在 Bob 系统上 (<B>)
                    # x = res[0, 0] - MA
                    idx_b_marg = bindex(res[1, 0], res[0, 0] - m_vec[0])
                    constraints.append(G[i, j] == cg[0, idx_b_marg])  # 第 0 行是 A 的常数项

                else:
                    # 测量在 Alice 系统上 (<A>)
                    idx_a_marg = aindex(res[1, 0], res[0, 0])
                    constraints.append(G[i, j] == cg[idx_a_marg, 0])  # 第 0 列是 B 的常数项

            # 4. 非交换项 (Non-commuting)
            else:  # res_type == -1
                # 检查之前是否遇到过这个 RES
                res_fnd = find_in_cell(res, res_catalog)

                if res_fnd == -1:
                    # 尝试查找其共轭/转置 (fliplr) 是否存在
                    # MATLAB: product_of_orthogonal(fliplr(res), m_vec)
                    res_conj, _ = product_of_orthogonal(res[:, ::-1], m_vec)
                    res_fnd = find_in_cell(res_conj, res_catalog)

                if res_fnd == -1:
                    # 如果仍然没找到，这是新的
                    res_catalog.append(res)
                    res_loc.append([i, j])
                else:
                    # 如果找到了，让当前 G(i,j) 等于之前记录位置的值
                    # 注意: res_loc 存储的是基于 0 的索引，直接使用
                    prev_loc = res_loc[res_fnd]
                    constraints.append(G[i, j] == G[prev_loc[0], prev_loc[1]])

    # --- 5. 求解问题 ---
    # 我们不仅检查可行性，还可以最小化 0 来检查约束满足情况
    # 或者对于 feasibility problem，CVXPY 通常留空 objective
    prob = cp.Problem(cp.Minimize(0), constraints)

    try:
        prob.solve(solver=cp.SCS, eps=tol)  # 使用 SCS 求解器，或其他已安装的求解器
    except cp.SolverError:
        print("Solver Error encountered.")
        return 0.0

    # 处理结果
    # 如果状态是 optimal，则满足 NPA
    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        return 1.0
    else:
        # INFEASIBLE 或 UNBOUNDED
        return 0.0