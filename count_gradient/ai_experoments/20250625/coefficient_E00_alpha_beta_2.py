from sympy import symbols, sqrt, N
from alpha_beta_Y_2_correlations import EA0, EA1, EB0, EB1, EA0B0, EA0B1, EA1B0, EA1B1

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
            + (s * (E01 + s * B1)) / ((1 + s * A0)**2 * sqrt(1 - nc2**2))
    print(f"A0: {N(term1)}")

    # Term 2
    term2 = - (t * (E10 + t * B0)) / ((1 + t * A1)**2 * sqrt(1 - nc3**2)) \
            - (t * (E11 + t * B1)) / ((1 + t * A1)**2 * sqrt(1 - nc4**2))
    print(f"A1: {N(term2)}")

    # Term 3
    term3 = (s / ((1 + s * A0) * sqrt(1 - nc1**2))) \
            + (t / ((1 + t * A1) * sqrt(1 - nc3**2)))
    print(f"B0: {N(term3)}")

    # Term 4
    term4 = - (s / ((1 + s * A0) * sqrt(1 - nc2**2))) \
            + (t / ((1 + t * A1) * sqrt(1 - nc4**2)))
    print(f"B1: {N(term4)}")

    # Term 5
    term5 = 1 / ((1 + s * A0) * sqrt(1 - nc1**2))
    print(f"E00: {N(term5)}")

    # Term 6
    term6 = -1 / ((1 + s * A0) * sqrt(1 - nc2**2))
    print(f"E01: {N(term6)}")

    # Term 7
    term7 = 1 / ((1 + t * A1) * sqrt(1 - nc3**2))
    print(f"E10: {N(term7)}")

    # Term 8
    term8 = 1 / ((1 + t * A1) * sqrt(1 - nc4**2))
    print(f"E11: {N(term8)}")

    return [N(term1), N(term2), N(term3), N(term4), N(term5), N(term6), N(term7), N(term8)]

# 参数定义
#
# A0 = 0.500000
# A1 = 0
# B0 = 0.3536
# B1 = -0.3536
#
# E00 = 0.7071
# E01 = -0.7071
# E10 = 0.6124
# E11 = 0.6124
#
# s = 1
# t = 1

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


