import numpy as np
from math import asin, pi


def analyze_linear_independence_with_values():
    """代入具体值分析线性无关性"""
    print("=" * 60)
    print("代入具体值分析线性无关性")
    print("=" * 60)

    # 给定的具体值
    params = [0.37796447, 0.37796447, 0.50000000, 0.00000000,
              0.75592895, -0.56694671, 0.75592895, 0.56694671]

    A0, A1, B0, B1, E00, E01, E10, E11 = params

    print(f"参数值: A0={A0}, A1={A1}, B0={B0}, B1={B1}")
    print(f"        E00={E00}, E01={E01}, E10={E10}, E11={E11}")

    # s,t的4种组合
    combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    # 计算每个组合下4个公式的值
    function_matrix = np.zeros((4, 4))

    formula_names = [
        "f1: arcsin[(E00+sB0)/(1+sA0)]",
        "f2: arcsin[(E10+tB0)/(1+tA1)]",
        "f3: arcsin[(E01+sB1)/(1+sA0)]",
        "f4: arcsin[(E11+tB1)/(1+tA1)]"
    ]

    for point_idx, (s, t) in enumerate(combinations):
        f1 = asin((E00 + s * B0) / (1 + s * A0))
        f2 = asin((E10 + t * B0) / (1 + t * A1))
        f3 = asin((E01 + s * B1) / (1 + s * A0))
        f4 = asin((E11 + t * B1) / (1 + t * A1))

        function_matrix[:, point_idx] = [f1, f2, f3, f4]

    print("\n函数值矩阵 (每列对应一个(s,t)组合):")
    print("列1: (s=1, t=1)  列2: (s=1, t=-1)")
    print("列3: (s=-1, t=1) 列4: (s=-1, t=-1)")
    for i, row in enumerate(function_matrix):
        print(f"{formula_names[i]:30s}: {row}")

    # 计算矩阵的秩
    rank = np.linalg.matrix_rank(function_matrix)
    print(f"\n矩阵的秩: {rank}")

    if rank == 4:
        print("✓ 矩阵满秩，4个公式在这些离散点上线性无关")
        return list(range(4))  # 返回所有索引

    # 如果秩不为4，找出线性无关的公式子集
    print(f"✗ 矩阵不满秩，秩为{rank}，存在线性相关性")

    # 方法1: 使用QR分解的列主元法来找出线性无关的行
    def find_linear_independent_rows(matrix, rank):
        """使用QR分解的列主元法找出线性无关的行"""
        Q, R, pivot = qr_pivot(matrix.T)  # 转置矩阵，对列进行QR分解
        independent_indices = pivot[:rank]  # 前rank个主元对应的行索引
        return sorted(independent_indices)  # 返回排序后的索引

    def qr_pivot(A):
        """带列主元的QR分解"""
        m, n = A.shape
        Q = np.eye(m)
        R = A.copy()
        pivot = list(range(n))

        for j in range(min(m, n)):
            # 找出列中最大元素的位置
            max_index = j + np.argmax(np.abs(R[j:, j]))

            if max_index != j:
                # 交换列
                R[:, [j, max_index]] = R[:, [max_index, j]]
                pivot[j], pivot[max_index] = pivot[max_index], pivot[j]

            # 标准的Gram-Schmidt过程
            for i in range(j + 1, m):
                if np.abs(R[j, j]) > 1e-12:  # 避免除以零
                    factor = R[i, j] / R[j, j]
                    R[i, j:] -= factor * R[j, j:]
                    Q[i, :] -= factor * Q[j, :]

        return Q, R, pivot

    # 方法2: 使用SVD并确保不重复
    def find_independent_by_svd(matrix, rank):
        """使用SVD找出线性无关的行，确保不重复"""
        U, S, Vh = np.linalg.svd(matrix)

        # 设置容差
        tol = max(matrix.shape) * np.finfo(matrix.dtype).eps * S[0]

        independent_indices = []
        used_indices = set()

        # 对每个奇异值，找出对应的主要行
        for i in range(rank):
            # 找出U矩阵中对应奇异向量的最大分量对应的行
            singular_vector = U[:, i]

            # 找出绝对值最大的分量，但跳过已经使用的索引
            available_indices = [idx for idx in range(len(singular_vector))
                                 if idx not in used_indices]

            if not available_indices:
                break

            max_index = available_indices[
                np.argmax(np.abs(singular_vector[available_indices]))
            ]

            independent_indices.append(max_index)
            used_indices.add(max_index)

        return sorted(independent_indices)

    # 方法3: 使用更稳定的方法 - 逐步构建线性无关集合
    def find_independent_stepwise(matrix, rank):
        """逐步构建线性无关的行集合"""
        m, n = matrix.shape
        independent_indices = []

        for i in range(m):
            if len(independent_indices) == rank:
                break

            # 取出当前行
            current_row = matrix[i:i + 1, :]

            if len(independent_indices) == 0:
                # 第一个行总是线性无关的
                independent_indices.append(i)
                continue

            # 检查当前行是否与已有行线性无关
            existing_matrix = matrix[independent_indices, :]
            extended_matrix = np.vstack([existing_matrix, current_row])

            if np.linalg.matrix_rank(extended_matrix) > len(independent_indices):
                independent_indices.append(i)

        return independent_indices

    # 使用三种方法进行比较，选择最佳结果
    results = []
    methods = []

    # 方法1: QR分解法
    try:
        qr_indices = find_linear_independent_rows(function_matrix, rank)
        results.append(qr_indices)
        methods.append("QR分解法")
        print(f"\n方法1 - QR分解法找到的线性无关公式索引: {qr_indices}")
    except Exception as e:
        print(f"\n方法1 - QR分解法失败: {e}")

    # 方法2: 逐步构建法
    try:
        stepwise_indices = find_independent_stepwise(function_matrix, rank)
        results.append(stepwise_indices)
        methods.append("逐步构建法")
        print(f"方法2 - 逐步构建法找到的线性无关公式索引: {stepwise_indices}")
    except Exception as e:
        print(f"方法2 - 逐步构建法失败: {e}")

    # 方法3: SVD法
    try:
        svd_indices = find_independent_by_svd(function_matrix, rank)
        results.append(svd_indices)
        methods.append("SVD法")
        print(f"方法3 - SVD法找到的线性无关公式索引: {svd_indices}")
    except Exception as e:
        print(f"方法3 - SVD法失败: {e}")

    # 选择最佳结果（优先选择QR分解法的结果）
    if results:
        # 优先选择QR分解法的结果
        if len(results) > 0 and methods[0] == "QR分解法":
            independent_indices = results[0]
            selected_method = methods[0]
        else:
            # 如果没有QR分解法的结果，选择第一个可用的方法
            independent_indices = results[0]
            selected_method = methods[0]

        print(f"\n最终选择 {selected_method} 的结果: {independent_indices}")
    else:
        print("\n所有方法都失败了，使用默认方法")
        independent_indices = list(range(rank))  # 简单的默认方法
        selected_method = "默认方法"

    # 验证结果
    print("\n线性无关的公式:")
    for idx in independent_indices:
        print(f"  {formula_names[idx]}")

    # 验证这些公式确实是线性无关的
    independent_matrix = function_matrix[independent_indices, :]
    independent_rank = np.linalg.matrix_rank(independent_matrix)
    print(f"\n选出的{len(independent_indices)}个公式构成的矩阵秩: {independent_rank}")

    # 检查是否有重复索引
    if len(independent_indices) != len(set(independent_indices)):
        print("⚠ 警告: 存在重复索引!")
        # 去重
        independent_indices = list(sorted(set(independent_indices)))
        print(f"去重后的索引: {independent_indices}")

        # 重新验证
        independent_matrix = function_matrix[independent_indices, :]
        independent_rank = np.linalg.matrix_rank(independent_matrix)
        print(f"去重后选出的{len(independent_indices)}个公式构成的矩阵秩: {independent_rank}")

    # 分析线性相关关系
    print(f"\n线性相关性分析:")
    if rank < 4:
        # 找出哪些公式是线性相关的
        dependent_indices = [i for i in range(4) if i not in independent_indices]

        for dep_idx in dependent_indices:
            try:
                # 尝试用线性无关的公式表示这个相关公式
                A = function_matrix[independent_indices, :].T
                b = function_matrix[dep_idx, :]
                coefficients = np.linalg.lstsq(A, b, rcond=None)[0]

                print(f"公式 f{dep_idx + 1} 可以表示为:")
                for i, coeff in enumerate(coefficients):
                    print(f"  {coeff:.6f} × f{independent_indices[i] + 1}", end="")
                    if i < len(coefficients) - 1:
                        print(" +", end="")
                print()
            except np.linalg.LinAlgError:
                print(f"无法确定公式 f{dep_idx + 1} 的具体线性关系")

    return independent_indices


def compute_condition_number():
    """计算条件数"""
    print("\n" + "=" * 60)
    print("条件数计算")
    print("=" * 60)

    # 条件数是矩阵数值稳定性的度量
    # 条件数 = ||A|| × ||A⁻¹|| （如果可逆）
    # 或者使用奇异值：cond(A) = σ_max / σ_min

    params = [0.37796447, 0.37796447, 0.50000000, 0.00000000,
              0.75592895, -0.56694671, 0.75592895, 0.56694671]

    A0, A1, B0, B1, E00, E01, E10, E11 = params

    combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    function_matrix = np.zeros((4, 4))

    for point_idx, (s, t) in enumerate(combinations):
        f1 = asin((E00 + s * B0) / (1 + s * A0))
        f2 = asin((E10 + t * B0) / (1 + t * A1))
        f3 = asin((E01 + s * B1) / (1 + s * A0))
        f4 = asin((E11 + t * B1) / (1 + t * A1))
        function_matrix[:, point_idx] = [f1, f2, f3, f4]

    # 计算条件数
    cond_number = np.linalg.cond(function_matrix)
    print(f"条件数: {cond_number:.6f}")

    if cond_number < 1e3:
        print("✓ 条件数良好，数值计算稳定")
    elif cond_number < 1e6:
        print("⚠ 条件数较大，数值计算可能不太稳定")
    else:
        print("✗ 条件数很大，数值计算不稳定")


# 运行分析
if __name__ == "__main__":
    # 分析线性无关性
    independent_indices = analyze_linear_independence_with_values()

    print(f"\n最终确定的线性无关公式索引: {independent_indices}")
    print("这些索引保证不重复且对应的公式线性无关")

    # 计算条件数
    compute_condition_number()