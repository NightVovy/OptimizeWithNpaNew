import numpy as np
from typing import List, Any, Tuple

def product_of_orthogonal(ind: np.ndarray, m_vec: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    确定由指标矩阵 IND 指定的测量算符乘积的性质，并递归地简化它。

    对应 MATLAB 函数: function [res,res_type] = product_of_orthogonal(ind,m_vec)

    参数:
        ind (np.ndarray): 2xk 的指标矩阵。
        m_vec (np.ndarray): [MA, MB]，测量设置数量。

    返回:
        Tuple[np.ndarray, int]: (简化后的指标矩阵 res, 结果类型 res_type)
        res_type 定义:
            -1: 无法完全简化 (包含非交换项)
             0: 零算符 (正交)
             1: 单一测量算符 (或恒等算符)
             2: 两个可交换的测量算符乘积
    """
    res_type = -1
    res = ind.copy()  # 对应 MATLAB: res = ind;

    # 辅助逻辑：判断某个测量属于 Alice 还是 Bob
    # MATLAB: min(floor(ind(1,i)/m_vec(1)),1) (用来得到 0 或 1)
    # Python: 如果 index >= m_vec[0] 则为 Bob (1)，否则为 Alice (0)
    ma = m_vec[0]

    def get_party(meas_idx):
        return 1 if meas_idx >= ma else 0

    while True:
        length = res.shape[1]  # 对应 MATLAB: len = size(ind,2);
        made_simplification = False  # 标记本轮是否发生了简化

        # --- 基本情况 1: 单一算符 ---
        # MATLAB: if(len == 1); res_type = 1; return;
        if length == 1:
            res_type = 1
            return res, res_type

        # --- 基本情况 2: 两个算符 ---
        # MATLAB: elseif(len == 2 ... )
        # 检查: 不是恒等算符(row 1 >= 0), 且属于不同方(Alice/Bob)
        if length == 2 and res[1, 0] >= 0 and res[1, 1] >= 0:
            party0 = get_party(res[0, 0])
            party1 = get_party(res[0, 1])

            # 如果属于不同方 (意味着可交换)
            if party0 != party1:
                # 规范排序: Alice 在前，Bob 在后
                # MATLAB: res = sortrows(res.').';
                if party0 > party1:  # 如果第一个是 Bob, 第二个是 Alice -> 交换
                    res = res[:, [1, 0]]
                res_type = 2
                return res, res_type

        # --- 复杂情况: 递归简化 ---
        # MATLAB: 嵌套循环 for i=1:len-1, for j=i+1:len
        # Python: 同样遍历，但一旦简化，修改 res 并跳出循环重新开始 (模拟 MATLAB 的递归返回)
        for i in range(length - 1):
            for j in range(i + 1, length):

                # 1. 正交性 (Orthogonality)
                # MATLAB: ind(1,i)==ind(1,j) && ind(2,i)~=ind(2,j) ...
                # 解释: 相同的测量设置 (row 0)，但结果不同 (row 1)，且都不是恒等算符 (-1)
                # 结果: 乘积为 0
                if (res[1, i] >= 0 and res[1, j] >= 0 and
                        res[0, i] == res[0, j] and res[1, i] != res[1, j]):
                    res_type = 0
                    # 返回一个特殊的标记表示 0 算符，通常用 [0; -1] 占位但类型为 0
                    return np.array([[0], [-1]], dtype=int), res_type

                # 2. 幂等性 (Idempotency)
                # MATLAB: ind(1,i)==ind(1,j) && ind(2,i)==ind(2,j)
                # 解释: 完全相同的算符相乘 A*A = A。合并它们。
                elif res[0, i] == res[0, j] and res[1, i] == res[1, j]:
                    # MATLAB: product_of_orthogonal(ind(:,[1:j-1,j+1:len]),...)
                    # Python: 删除第 j 列
                    res = np.delete(res, j, axis=1)
                    made_simplification = True
                    break  # 跳出内层循环，重新 while True

                # 3. 恒等算符 (Identity) - 第一项是 I
                # MATLAB: ind(2,i) == -1
                elif res[1, i] == -1:
                    res = np.delete(res, i, axis=1)
                    made_simplification = True
                    break

                # 4. 恒等算符 (Identity) - 第二项是 I
                # MATLAB: ind(2,j) == -1
                elif res[1, j] == -1:
                    res = np.delete(res, j, axis=1)
                    made_simplification = True
                    break

                # 边界检查: 如果删除后变空了 (例如 [I, I] -> []), 重置为恒等算符
                if res.shape[1] == 0:
                    res = np.array([[0], [-1]], dtype=int)
                    made_simplification = True
                    break

                # 5. 同方阻隔 (Same party barrier)
                # MATLAB: min(floor...i) == min(floor...j) -> break
                # 解释: 如果 i 和 j 属于同一方 (且不是上述的正交/幂等情况)，
                # 则它们是非交换的，且中间不能跨越。停止增加 j，继续下一个 i。
                if get_party(res[0, i]) == get_party(res[0, j]):
                    break

                    # 6. 交换顺序 (Commutativity)
                # MATLAB: ind(1,i) > ind(1,j) ...
                # 解释: 属于不同方 (因为没在上面 break)，且 i 的设置索引 > j 的设置索引。
                # 这意味着 Bob 在 Alice 前面。利用对易性交换它们。
                elif res[0, i] > res[0, j]:  # 这里简单用索引比较，因为 Bob 索引总是 > Alice
                    # 交换列 i 和 j
                    # MATLAB: ind(:,[1:i-1,j,i+1:j-1,i,j+1:len])
                    cols = list(range(length))
                    cols[i], cols[j] = cols[j], cols[i]
                    res = res[:, cols]
                    made_simplification = True
                    break

            if made_simplification:
                break  # 跳出外层循环，重新 while True

        # 如果遍历所有组合都没有发生任何简化，说明无法进一步简化了
        if not made_simplification:
            break

    # --- 最终状态检查 ---
    # 循环结束后，res 已经是简化形式。再次检查是否落入基本情况。
    length = res.shape[1]
    if length == 1:
        res_type = 1
    elif (length == 2 and res[1, 0] >= 0 and res[1, 1] >= 0 and
          get_party(res[0, 0]) != get_party(res[0, 1])):
        res_type = 2
    else:
        # 既不是 0，也不是 1 或 2，说明是复杂的非交换项
        res_type = -1

    return res, res_type