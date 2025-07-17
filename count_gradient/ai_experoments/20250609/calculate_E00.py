from sympy import symbols, sqrt, N

def calculate_terms(E00, E01, E10, E11, A0, A1, B0, B1, s, t):

    # 方块关联函数
    def new_corre(Exy, Ax, By, s):
        return (Exy + s * By) / (1 + s * Ax)

    # 计算 new_corre 的值
    nc1 = new_corre(E00, A0, B0, s)
    nc2 = new_corre(E01, A0, B1, s)
    nc3 = new_corre(E10, A1, B0, t)
    nc4 = new_corre(E11, A1, B1, t)

    # 打印 new_corre 的值
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

E00 = 0.755929
E01 = -0.755929
E10 = 0.566947
E11 = 0.566947
A0 = 0.500000
A1 = 0
B0 = 0.377964
B1 = -0.377964
s = 1
t = 1

# 执行函数
results = calculate_terms(E00, E01, E10, E11, A0, A1, B0, B1, s, t)
