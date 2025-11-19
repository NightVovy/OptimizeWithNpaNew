import numpy as np
import cvxpy as cp
from typing import List, Tuple, Union, Any

"""
此段代码的逻辑完全符合QETLAB的NPAHierarchy.m

包含以下函数：
    主函数 IS_NPA = NPAHierarchy(CG, DESC, K) 。返回0或1.
    判定给定关联矩阵(maybe moment matrix)是否满足NPA层析条件。
    参数： 
        cg：关联矩阵的（Collins-Gisin格式）
        desc：ALICE和BOB的输入输出数量
        K：NPA层数，如果是1+ab记得写成'1+ab'
        
    辅助函数 find_in_cell(val,A)
    
    辅助函数 update_ind(old_ind,k,m_vec,o_vec)
    
    辅助函数 product_of_orthogonal(ind,m_vec)

"""

def NPAHierarchy(cg: np.ndarray, desc: List[int], k: Union[int, str] = 1) -> float:
    """
    判定给定关联矩阵是否满足NPA层级条件（Python移植版）

    参数:
        cg: 关联矩阵（Collins-Gisin格式）
        desc: 场景描述 [OA, OB, MA, MB]，分别为Alice/Bob的测量结果数和测量设置数
        k: NPA层级（整数或字符串，如'1+ab+aab'）

    返回:
        is_npa: 1表示满足NPA条件，0表示不满足（考虑数值精度）
    """
    # 解析场景参数
    desc = np.array(desc).flatten()
    o_vec = desc[:2]  # OA, OB
    m_vec = desc[2:]  # MA, MB
    tol = np.finfo(float).eps ** (3 / 4)  # 数值 tolerance

    # 解析K参数（处理整数或字符串格式）
    base_k, num_k_compon = 0, 1
    if isinstance(k, int):
        base_k = k
    elif isinstance(k, str):
        k_types = k.lower().split('+')
        base_k = int(k_types[0])
        num_k_compon = len(k_types)
        k_val = base_k
        for j in range(1, num_k_compon):
            k_val = max(k_val, len(k_types[j]))
        k = k_val
    else:
        raise ValueError("K must be a positive integer or a string.")

    # 匿名函数替代MATLAB的aindex和bindex（索引计算）
    def aindex(a: int, x: int) -> int:
        return 2 + a + x * (o_vec[0] - 1)  # MATLAB索引从1开始，Python需注意偏移

    def bindex(b: int, y: int) -> int:
        return 2 + b + y * (o_vec[1] - 1)

    # 计算G的维度上限（与MATLAB逻辑一致）
    tot_dim = 1 + ((o_vec - 1) @ m_vec) ** k

    # 生成所有有效的测量算子乘积（ind_catalog）
    ind_catalog = []
    j_ind = np.vstack([np.zeros(k), -np.ones(k)])  # 初始索引矩阵

    for _ in range(tot_dim):
        res, res_type = product_of_orthogonal(j_ind, m_vec)
        # 检查是否为新的且有效的算子
        res_fnd = find_in_cell(res, ind_catalog)
        if res_fnd == 0 and res_type != 0:
            is_valid_res = (res.shape[1] <= base_k)
            if not is_valid_res and num_k_compon >= 2:
                num_a_res = sum(res[0, :] < m_vec[0])
                num_b_res = res.shape[1] - num_a_res
                for i in range(1, num_k_compon):
                    num_a_fnd = k_types[i].count('a')
                    num_b_fnd = k_types[i].count('b')
                    if num_a_res <= num_a_fnd and num_b_res <= num_b_fnd:
                        is_valid_res = True
                        break
            if is_valid_res:
                ind_catalog.append(res)
        # 更新索引矩阵
        j_ind = update_ind(j_ind, k, m_vec, o_vec - 1)

    real_dim = len(ind_catalog)
    if real_dim == 0:
        return 0.0  # 无有效算子，直接返回不可行

    # 构建SDP问题（cvxpy）
    G = cp.Variable((real_dim, real_dim), symmetric=True)
    constraints = [G >> 0]  # 半正定约束

    res_catalog = []
    res_loc = []

    # 添加元素约束（对应MATLAB的双重循环）
    for i in range(real_dim):
        for j in range(i, real_dim):
            # 计算算子乘积类型
            combined_ind = np.hstack([np.fliplr(ind_catalog[i]), ind_catalog[j]])
            res, res_type = product_of_orthogonal(combined_ind, m_vec)

            if res_type == 0:
                # 约束为0
                constraints.append(G[i, j] == 0)
            elif res_type == 2:
                # 对应cg矩阵的元素
                a_idx = aindex(res[1, 0], res[0, 0]) - 1  # Python索引从0开始，减1
                b_idx = bindex(res[1, 1], res[0, 1] - m_vec[0]) - 1
                constraints.append(G[i, j] == cg[a_idx, b_idx])
            elif res_type == 1:
                # 单位算子或边缘分布
                if np.array_equal(res, np.array([[0], [-1]])):
                    constraints.append(G[i, j] == 1)
                elif res[0, 0] >= m_vec[0]:
                    # Bob的测量
                    b_idx = bindex(res[1, 0], res[0, 0] - m_vec[0]) - 1
                    constraints.append(G[i, j] == cg[0, b_idx])  # cg[1, ...]在MATLAB中对应0索引
                else:
                    # Alice的测量
                    a_idx = aindex(res[1, 0], res[0, 0]) - 1
                    constraints.append(G[i, j] == cg[a_idx, 0])
            else:  # res_type == -1
                # 处理非交换算子的等价约束
                res_fnd = find_in_cell(res, res_catalog)
                if res_fnd == 0:
                    res_flipped, _ = product_of_orthogonal(np.fliplr(res), m_vec)
                    res_fnd = find_in_cell(res_flipped, res_catalog)
                if res_fnd == 0:
                    res_catalog.append(res)
                    res_loc.append((i, j))
                else:
                    # 约束当前元素与之前的等价元素相等
                    i_prev, j_prev = res_loc[res_fnd - 1]  # res_fnd是1-based索引
                    constraints.append(G[i, j] == G[i_prev, j_prev])

    # 求解可行性问题（无目标函数）
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.MOSEK, eps=tol)  # 用SCS求解器，可替换为其他支持SDP的求解器

    # 计算is_npa（参考MATLAB逻辑）
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        # 可行性问题最优值为0，若求解成功则认为可行
        is_npa = 1.0 - min(prob.value, 1.0) if prob.value is not None else 1.0
    else:
        is_npa = 0.0

    # 处理非CVX变量输入时的数值优化
    if not isinstance(cg, cp.Variable):
        is_npa = round(is_npa)
        # 状态提示（对应MATLAB的警告/错误）
        if prob.status == cp.OPTIMAL_INACCURATE:
            print("警告：CVXPY遇到轻微数值问题，考虑调整 tolerance。")
        elif prob.status in [cp.INFEASIBLE_INACCURATE, cp.UNBOUNDED_INACCURATE]:
            print("警告：CVXPY遇到数值问题，结果可能不准确。")
        elif prob.status in [cp.INFEASIBLE, cp.UNBOUNDED]:
            raise RuntimeError(f"数值求解失败（状态：{prob.status}），请调整 tolerance。")

    return is_npa


# 辅助函数：查找元素是否在列表中（替代MATLAB的find_in_cell）
def find_in_cell(val: np.ndarray, cell_list: List[np.ndarray]) -> int:
    for i, item in enumerate(cell_list):
        if np.array_equal(item, val):
            return i + 1  # 返回1-based索引，与MATLAB一致
    return 0


# 辅助函数：更新索引矩阵（对应MATLAB的update_ind）
def update_ind(old_ind: np.ndarray, k: int, m_vec: np.ndarray, o_vec: np.ndarray) -> np.ndarray:
    # 若包含-1（单位算子），则初始化为全0
    if np.min(old_ind) == -1:
        return np.zeros((2, k), dtype=int)

    new_ind = old_ind.copy()
    new_ind[1, k - 1] += 1  # Python索引从0开始，最后一列是k-1

    # 处理"里程表"进位逻辑
    for l in range(k - 1, -1, -1):
        # 检查结果索引是否超过上限
        meas_idx = new_ind[0, l]
        party = min(meas_idx // m_vec[0], 1)  # 0: Alice, 1: Bob
        if new_ind[1, l] >= o_vec[party]:
            new_ind[1, l] = 0
            new_ind[0, l] += 1
        else:
            return new_ind

        # 检查测量索引是否超过上限
        if new_ind[0, l] >= sum(m_vec):
            new_ind[0, l] = 0
            if l >= 1:
                new_ind[1, l - 1] += 1
            else:
                return new_ind
        else:
            return new_ind
    return new_ind


# 辅助函数：处理测量算子乘积（对应MATLAB的product_of_orthogonal）
def product_of_orthogonal(ind: np.ndarray, m_vec: np.ndarray) -> Tuple[np.ndarray, int]:
    res = ind.copy()
    res_type = -1
    len_ind = res.shape[1]

    # 单个测量算子
    if len_ind == 1:
        res_type = 1
        return res, res_type

    # 两个可交换的非单位算子
    if len_ind == 2:
        if res[1, 0] >= 0 and res[1, 1] >= 0:
            party1 = min(res[0, 0] // m_vec[0], 1)
            party2 = min(res[0, 1] // m_vec[0], 1)
            if party1 != party2:
                # 按Alice在前、Bob在后排序
                if res[0, 0] > res[0, 1]:
                    res = res[:, [1, 0]]
                res_type = 2
                return res, res_type

    # 复杂情况：递归简化
    for i in range(len_ind - 1):
        for j in range(i + 1, len_ind):
            # 正交算子（乘积为0）
            if (res[1, i] >= 0 and res[1, j] >= 0 and
                    res[0, i] == res[0, j] and res[1, i] != res[1, j]):
                return res, 0
            # 相同算子（合并）
            elif res[0, i] == res[0, j] and res[1, i] == res[1, j]:
                new_ind = np.hstack([res[:, :j], res[:, j + 1:]])
                return product_of_orthogonal(new_ind, m_vec)
            # 包含单位算子（合并）
            elif res[1, i] == -1:
                new_ind = np.hstack([res[:, :i], res[:, i + 1:]])
                return product_of_orthogonal(new_ind, m_vec)
            elif res[1, j] == -1:
                new_ind = np.hstack([res[:, :j], res[:, j + 1:]])
                return product_of_orthogonal(new_ind, m_vec)
            # 同方非正交算子（停止当前循环）
            party_i = min(res[0, i] // m_vec[0], 1)
            party_j = min(res[0, j] // m_vec[0], 1)
            if party_i == party_j:
                break
            # 不同方算子（交换顺序使Alice在前）
            elif res[0, i] > res[0, j]:
                # 交换i和j列
                new_ind = res.copy()
                new_ind[:, [i, j]] = new_ind[:, [j, i]]
                return product_of_orthogonal(new_ind, m_vec)

    return res, res_type