import numpy as np
from sympy import symbols, sqrt, N
from alpha_matrix_eigrnvalue import EA0, EA1, EB0, EB1, EA0B0, EA0B1, EA1B0, EA1B1, matrices

def calculate_terms(A0, A1, B0, B1, E00, E01, E10, E11, s, t):

    # 方块关联函数
    def new_corre(Exy, Ax, By, s):
        return (Exy + s * By) / (1 + s * Ax)

    # 计算 new_corre 的值
    nc1 = new_corre(E00, A0, B0, s)
    nc2 = new_corre(E01, A0, B1, s)
    nc3 = new_corre(E10, A1, B0, t)
    nc4 = new_corre(E11, A1, B1, t)

    # 打印 new_corre 的值
    print("\n\n")  # 这里加入两个换行符
    print("new_corre 值：")
    print(f"nc1 (E00, A0, B0, s): {N(nc1)}")
    print(f"nc2 (E01, A0, B1, s): {N(nc2)}")
    print(f"nc3 (E10, A1, B0, t): {N(nc3)}")
    print(f"nc4 (E11, A1, B1, t): {N(nc4)}")

    # Term 1
    term1 = - (s * (E00 + s * B0)) / ((1 + s * A0)**2 * sqrt(1 - nc1**2)) \
            - (s * (E01 + s * B1)) / ((1 + s * A0)**2 * sqrt(1 - nc2**2))
    print(f"A0: {N(term1)}")

    # Term 2
    term2 = - (t * (E10 + t * B0)) / ((1 + t * A1)**2 * sqrt(1 - nc3**2)) \
            + (t * (E11 + t * B1)) / ((1 + t * A1)**2 * sqrt(1 - nc4**2))
    print(f"A1: {N(term2)}")

    # Term 3
    term3 = (s / ((1 + s * A0) * sqrt(1 - nc1**2))) \
            + (t / ((1 + t * A1) * sqrt(1 - nc3**2)))
    print(f"B0: {N(term3)}")

    # Term 4
    term4 = (s / ((1 + s * A0) * sqrt(1 - nc2**2))) \
            - (t / ((1 + t * A1) * sqrt(1 - nc4**2)))
    print(f"B1: {N(term4)}")

    # Term 5
    term5 = 1 / ((1 + s * A0) * sqrt(1 - nc1**2))
    print(f"E00: {N(term5)}")

    # Term 6
    term6 = 1 / ((1 + s * A0) * sqrt(1 - nc2**2))
    print(f"E01: {N(term6)}")

    # Term 7
    term7 = 1 / ((1 + t * A1) * sqrt(1 - nc3**2))
    print(f"E10: {N(term7)}")

    # Term 8
    term8 = - 1 / ((1 + t * A1) * sqrt(1 - nc4**2))
    print(f"E11: {N(term8)}")

    return [N(term1), N(term2), N(term3), N(term4), N(term5), N(term6), N(term7), N(term8)]


# 使用获取的期望值作为输入参数
results = calculate_terms(
    A0=EA0,
    A1=EA1,
    B0=EB0,
    B1=EB1,
    E00=EA0B0,
    E01=EA0B1,
    E10=EA1B0,
    E11=EA1B1,
    s = 1,
    t = 1
)


def calculate_combined_matrix(terms, matrices):
    """
    计算 term1*A0 + term2*A1 + ... + term8*A1B1 的组合矩阵
    参数:
        terms: 8个系数的列表 [term1, term2, ..., term8]
        matrices: 包含8个 NumPy 数组的字典（键为 'A0', 'A1', ..., 'A1B1'）
    返回:
        combined_matrix: 组合后的 4x4 复数矩阵
    """
    # 初始化组合矩阵（明确指定复数类型）
    combined_matrix = np.zeros((4, 4), dtype=np.complex128)

    # 按顺序匹配 terms 和 matrices 的键
    matrix_keys = ['A0', 'A1', 'B0', 'B1', 'A0B0', 'A0B1', 'A1B0', 'A1B1']

    # 逐项相加（确保 terms 是数值类型）
    for term, key in zip(terms, matrix_keys):
        combined_matrix += complex(term) * matrices[key]  # 直接使用 matrices 中的 NumPy 数组

    return combined_matrix


# 主程序
if __name__ == "__main__":
    # 假设已经从原代码获取了以下变量:
    # EA0, EA1, EB0, EB1, EA0B0, EA0B1, EA1B0, EA1B1
    # 和 matrices 字典

    # 计算各项系数
    terms = calculate_terms(
        A0=EA0,
        A1=EA1,
        B0=EB0,
        B1=EB1,
        E00=EA0B0,
        E01=EA0B1,
        E10=EA1B0,
        E11=EA1B1,
        s=1,
        t=1
    )

    # 打印各项系数
    print("\n计算得到的各项系数:")
    print(f"term1 (A0系数): {terms[0]:.6f}")
    print(f"term2 (A1系数): {terms[1]:.6f}")
    print(f"term3 (B0系数): {terms[2]:.6f}")
    print(f"term4 (B1系数): {terms[3]:.6f}")
    print(f"term5 (E00系数): {terms[4]:.6f}")
    print(f"term6 (E01系数): {terms[5]:.6f}")
    print(f"term7 (E10系数): {terms[6]:.6f}")
    print(f"term8 (E11系数): {terms[7]:.6f}")

    # 计算组合矩阵
    combined_matrix = calculate_combined_matrix(terms, matrices)

    # 计算最大特征值
    eigenvalues = np.linalg.eigvals(combined_matrix)
    max_eigenvalue = np.max(eigenvalues.real)  # 取实部

    # 输出结果
    print("\n新bell表达式的组合矩阵:")
    print(combined_matrix)

    print("\n新bell表达式的组合矩阵的最大特征值:")
    print(f"{max_eigenvalue:.6f}")