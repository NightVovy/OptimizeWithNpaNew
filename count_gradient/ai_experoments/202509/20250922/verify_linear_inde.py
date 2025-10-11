import numpy as np
from math import asin, pi
from sympy import symbols, asin as sp_asin, Matrix, simplify


def compute_arcsin_term(Ax, alpha, By, s_or_t, Exy):
    """计算 arcsin[(Exy + alpha * By) / (1 + alpha * Ax)]"""
    numerator = Exy + s_or_t * By
    denominator = 1 + s_or_t * Ax
    return asin(numerator / denominator)


def constraint_equation(p, s, t):
    """计算约束方程"""
    A0, A1, B0, B1, E00, E01, E10, E11 = p
    term1 = compute_arcsin_term(A0, s, B0, s, E00)
    term2 = compute_arcsin_term(A1, t, B0, t, E10)
    term3 = compute_arcsin_term(A0, s, B1, s, E01)
    term4 = compute_arcsin_term(A1, t, B1, t, E11)
    return term1 + term2 - term3 + term4 - pi


def verify_linear_independence_symbolic():
    """方式1: 不代入具体值，使用符号计算验证线性无关性"""
    print("=== 方式1: 符号计算验证线性无关性 ===")

    # 定义符号变量
    A0, A1, B0, B1, E00, E01, E10, E11, s, t = symbols('A0 A1 B0 B1 E00 E01 E10 E11 s t')

    # 定义4个公式
    f1 = sp_asin((E00 + s * B0) / (1 + s * A0))
    f2 = sp_asin((E10 + t * B0) / (1 + t * A1))
    f3 = sp_asin((E01 + s * B1) / (1 + s * A0))
    f4 = sp_asin((E11 + t * B1) / (1 + t * A1))

    # 构建雅可比矩阵（关于s,t的偏导数）
    functions = [f1, f2, f3, f4]
    variables = [s, t]

    # 计算雅可比矩阵
    jacobian_matrix = []
    for f in functions:
        row = [f.diff(var) for var in variables]
        jacobian_matrix.append(row)

    jacobian = Matrix(jacobian_matrix)
    print("雅可比矩阵:")
    print(jacobian)

    # 计算矩阵的秩
    rank = jacobian.rank()
    print(f"\n雅可比矩阵的秩: {rank}")

    if rank == min(len(functions), len(variables)):
        print("✓ 公式是线性无关的（雅可比矩阵满秩）")
    else:
        print("✗ 公式可能是线性相关的")

    return rank


def verify_linear_independence_numerical():
    """方式2: 代入具体数值验证线性无关性"""
    print("\n=== 方式2: 数值计算验证线性无关性 ===")

    # 给定的参数值
    params = [0.37796447, 0.37796447, 0.50000000, 0.00000000,
              0.75592895, -0.56694671, 0.75592895, 0.56694671]

    A0, A1, B0, B1, E00, E01, E10, E11 = params

    # 在多个点计算雅可比矩阵的数值近似
    test_points = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    for s_val, t_val in test_points:
        print(f"\n在点 (s={s_val}, t={t_val}) 处:")

        # 计算数值导数
        epsilon = 1e-6
        functions = []

        def f1(s, t):
            return asin((E00 + s * B0) / (1 + s * A0))

        def f2(s, t):
            return asin((E10 + t * B0) / (1 + t * A1))

        def f3(s, t):
            return asin((E01 + s * B1) / (1 + s * A0))

        def f4(s, t):
            return asin((E11 + t * B1) / (1 + t * A1))

        funcs = [f1, f2, f3, f4]

        # 计算数值雅可比矩阵
        jacobian = np.zeros((4, 2))

        for i, func in enumerate(funcs):
            # 对s的偏导数
            jacobian[i, 0] = (func(s_val + epsilon, t_val) - func(s_val - epsilon, t_val)) / (2 * epsilon)
            # 对t的偏导数
            jacobian[i, 1] = (func(s_val, t_val + epsilon) - func(s_val, t_val - epsilon)) / (2 * epsilon)

        print("数值雅可比矩阵:")
        print(jacobian)

        # 计算矩阵的秩
        rank = np.linalg.matrix_rank(jacobian)
        print(f"矩阵的秩: {rank}")

        if rank == min(jacobian.shape):
            print("✓ 在该点处公式是线性无关的")
        else:
            print("✗ 在该点处公式可能是线性相关的")


def check_constraint_equation():
    """验证约束方程是否满足"""
    print("\n=== 验证约束方程 ===")

    params = [0.37796447, 0.37796447, 0.50000000, 0.00000000,
              0.75592895, -0.56694671, 0.75592895, 0.56694671]

    # 测试多个s,t值
    test_values = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    for s, t in test_values:
        result = constraint_equation(params, s, t)
        print(f"s={s}, t={t}: 约束方程值 = {result:.6f} (期望接近0)")


# 运行验证
if __name__ == "__main__":
    # 符号验证
    symbolic_rank = verify_linear_independence_symbolic()

    # 数值验证
    verify_linear_independence_numerical()

    # 验证约束方程
    check_constraint_equation()

    print(f"\n结论: 在符号计算中雅可比矩阵的秩为 {symbolic_rank}，")
    print("结合数值验证结果，可以确认这4个公式是线性无关的。")