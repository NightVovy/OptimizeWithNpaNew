import numpy as np
from typing import List, Any, Tuple


def find_in_cell(val: np.ndarray, a: List[np.ndarray]) -> int:
    """
    在 numpy 数组的列表中查找给定的 numpy 数组。

    参数:
        val (np.ndarray): 要查找的目标数组。
        a (List[np.ndarray]): 包含 numpy 数组的列表（目录）。

    返回:
        int: 如果找到，返回基于 0 的索引 (0, 1, 2, ...)；否则返回 -1。
    """
    for i, item in enumerate(a):
        # 比较两个 numpy 数组是否相等
        if np.array_equal(item, val):
            return i  # 返回基于 0 的索引
    return -1


def update_ind(old_ind: np.ndarray, k: int, m_vec: np.ndarray, o_vec: np.ndarray) -> np.ndarray:
    """
    根据旧的指标矩阵生成下一个测量指标矩阵 (2 x k)。

    参数:
        old_ind (np.ndarray): 旧的指标矩阵 (2 x k)。
        k (int): 测量产品中的运算符数量上限。
        m_vec (np.ndarray): [MA, MB]，Alice 和 Bob 的测量设置数量。
        o_vec (np.ndarray): [OA-1, OB-1]，Alice 和 Bob 的结果数量减 1。

    返回:
        np.ndarray: 下一个指标矩阵 (2 x k)。
    """
    new_ind = old_ind.copy()

    # 检查是否是初始的恒等测量矩阵 [zeros(1,k);-ones(1,k)] 第一行全0第二行全-1
    if np.min(new_ind) == -1:
        # 如果是，转到第一个非恒等测量 [0...0; 0...0]
        return np.zeros((2, k), dtype=int)

    # 1. 增加最后一个结果索引 (Python 索引 k-1)
    new_ind[1, k - 1] += 1

    # 2. 模拟“里程表”操作，从右向左进位
    for l in range(k - 1, -1, -1):
        # 确定当前测量设置是属于 Alice (0) 还是 Bob (1)
        party_idx = int(new_ind[0, l] >= m_vec[0])

        # 确定该测量设置的结果上限 (o_vec[party_idx] = OA-1 或 OB-1)
        outcome_limit = o_vec[party_idx]

        # 检查是否结果索引达到上限 (需要进位到测量设置)
        if new_ind[1, l] > outcome_limit:
            new_ind[1, l] = 0
            new_ind[0, l] += 1  # 进位：测量设置索引增加
        else:
            return new_ind  # 结果索引未溢出，返回

        # 检查是否测量设置索引达到上限 (需要进位到前一个运算符)
        if new_ind[0, l] >= np.sum(m_vec):
            new_ind[0, l] = 0
            if l >= 1:
                # 进位到前一个运算符的结果索引 (Python 索引 l-1)
                new_ind[1, l - 1] += 1
            else:
                # l=0 且溢出，结束
                return new_ind
        else:
            return new_ind  # 测量设置索引未溢出，返回

    return new_ind