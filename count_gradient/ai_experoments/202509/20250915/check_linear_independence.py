import sympy as sp
import numpy as np


def check_linear_independence():
    # 定义符号变量
    theta, a0, a1, b0, b1 = sp.symbols('theta a0 a1 b0 b1')

    # 定义5个向量
    # T1 = [
    #     sp.sin(2 * theta) * sp.cos(a0),
    #     sp.sin(2 * theta) * sp.cos(a1),
    #     sp.sin(2 * theta) * sp.cos(b0),
    #     sp.sin(2 * theta) * sp.cos(b1),
    #     -sp.sin(a0) * sp.sin(b0) * sp.cos(2 * theta),
    #     -sp.sin(a0) * sp.sin(b1) * sp.cos(2 * theta),
    #     -sp.sin(a1) * sp.sin(b0) * sp.cos(2 * theta),
    #     -sp.sin(a1) * sp.sin(b1) * sp.cos(2 * theta)
    # ]
    T1 = [ # theta
        -2 * sp.sin(2 * theta) * sp.cos(a0),
        -2 * sp.sin(2 * theta) * sp.cos(a1),
        -2 * sp.sin(2 * theta) * sp.cos(b0),
        -2 * sp.sin(2 * theta) * sp.cos(b1),
        2 * sp.sin(a0) * sp.sin(b0) * sp.cos(2 * theta),
        2 * sp.sin(a0) * sp.sin(b1) * sp.cos(2 * theta),
        2 * sp.sin(a1) * sp.sin(b0) * sp.cos(2 * theta),
        2 * sp.sin(a1) * sp.sin(b1) * sp.cos(2 * theta)
    ]

    T2 = [
        sp.sin(a0) * sp.sin(theta),
        sp.sin(a1) * sp.sin(theta),
        sp.sin(b0) * sp.sin(theta),
        sp.sin(b1) * sp.sin(theta),
        sp.cos(a0) * sp.sin(b0) * sp.cos(theta) - sp.sin(a0) * sp.cos(b0) * sp.sin(theta),
        sp.cos(a0) * sp.sin(b1) * sp.cos(theta) - sp.sin(a0) * sp.cos(b1) * sp.sin(theta),
        sp.cos(a1) * sp.sin(b0) * sp.cos(theta) - sp.sin(a1) * sp.cos(b0) * sp.sin(theta),
        sp.cos(a1) * sp.sin(b1) * sp.cos(theta) - sp.sin(a1) * sp.cos(b1) * sp.sin(theta)
    ]

    T3 = [
        sp.sin(a0) * sp.cos(theta),
        sp.sin(a1) * sp.cos(theta),
        sp.sin(b0) * sp.cos(theta),
        sp.sin(b1) * sp.cos(theta),
        -sp.cos(a0) * sp.sin(b0) * sp.sin(theta) + sp.sin(a0) * sp.cos(b0) * sp.cos(theta),
        -sp.cos(a0) * sp.sin(b1) * sp.sin(theta) + sp.sin(a0) * sp.cos(b1) * sp.cos(theta),
        -sp.cos(a1) * sp.sin(b0) * sp.sin(theta) + sp.sin(a1) * sp.cos(b0) * sp.cos(theta),
        -sp.cos(a1) * sp.sin(b1) * sp.sin(theta) + sp.sin(a1) * sp.cos(b1) * sp.cos(theta)
    ]

    pd_a0 = [
        -sp.cos(2 * theta) * sp.sin(a0),
        0,
        0,
        0,
        -sp.sin(a0) * sp.cos(b0) + sp.sin(2 * theta) * sp.cos(a0) * sp.sin(b0),
        -sp.sin(a0) * sp.cos(b1) + sp.sin(2 * theta) * sp.cos(a0) * sp.sin(b1),
        0,
        0
    ]

    pd_b0 = [
        0,
        0,
        -sp.cos(2 * theta) * sp.sin(b0),
        0,
        -sp.cos(a0) * sp.sin(b0) + sp.sin(2 * theta) * sp.sin(a0) * sp.cos(b0),
        0,
        -sp.cos(a1) * sp.sin(b0) + sp.sin(2 * theta) * sp.sin(a1) * sp.cos(b0),
        0
    ]

    # 创建符号矩阵（每个向量作为一行）
    vectors = [T1, T2, T3, pd_a0, pd_b0]
    matrix_A = sp.Matrix(vectors)

    print("符号矩阵A:")
    sp.pprint(matrix_A)

    # 计算矩阵的秩
    print("\n计算矩阵的秩...")
    rank = matrix_A.rank()
    print(f"矩阵的秩: {rank}")
    print(f"向量个数: {len(vectors)}")

    if rank == len(vectors):
        print("5个向量线性无关")
        return True
    else:
        print("5个向量线性相关")
        return False


# 运行验证
if __name__ == "__main__":
    is_independent = check_linear_independence()

    if is_independent:
        print("结论: 5个向量是线性无关的")
    else:
        print("结论: 5个向量是线性相关的")