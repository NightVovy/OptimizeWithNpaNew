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
        return

    # 如果秩不为4，找出线性无关的公式子集
    print(f"✗ 矩阵不满秩，秩为{rank}，存在线性相关性")

    # 使用SVD找出线性无关的行
    U, S, Vh = np.linalg.svd(function_matrix)

    # 找出非零奇异值对应的行
    tol = max(function_matrix.shape) * np.finfo(function_matrix.dtype).eps * S[0]
    independent_indices = []

    for i in range(rank):
        # 找出每个奇异值对应的主要行
        max_index = np.argmax(np.abs(U[:, i]))
        independent_indices.append(max_index)

    print(f"\n线性无关的公式索引: {independent_indices}")
    print("线性无关的公式:")
    for idx in independent_indices:
        print(f"  {formula_names[idx]}")

    # 验证这些公式确实是线性无关的
    independent_matrix = function_matrix[independent_indices, :]
    independent_rank = np.linalg.matrix_rank(independent_matrix)
    print(f"\n选出的{len(independent_indices)}个公式构成的矩阵秩: {independent_rank}")

    # 分析线性相关关系
    print(f"\n线性相关性分析:")
    if rank == 3:
        # 找出哪个公式可以表示为其他三个的线性组合
        try:
            # 解线性方程组找出系数
            for target_idx in range(4):
                if target_idx not in independent_indices:
                    # 这个公式是线性相关的
                    A = function_matrix[independent_indices, :].T
                    b = function_matrix[target_idx, :]
                    coefficients = np.linalg.lstsq(A, b, rcond=None)[0]

                    print(f"公式 f{target_idx + 1} 可以表示为:")
                    for i, coeff in enumerate(coefficients):
                        print(f"  {coeff:.6f} × f{independent_indices[i] + 1}", end="")
                        if i < len(coefficients) - 1:
                            print(" +", end="")
                    print()
                    break

        except np.linalg.LinAlgError:
            print("无法确定具体的线性关系")


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
    analyze_linear_independence_with_values()

    # 计算条件数
    compute_condition_number()