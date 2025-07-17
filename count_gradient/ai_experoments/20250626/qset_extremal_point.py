import math


def check_equation(s, t, A0, B0, A1, B1, E00, E01, E10, E11):
    # 计算四个arcsin项
    term1 = math.asin((E00 + s * B0) / (1 + s * A0))
    term2 = math.asin((E10 + t * B0) / (1 + t * A1))
    term3 = math.asin((E01 + s * B1) / (1 + s * A0))
    term4 = math.asin((E11 + t * B1) / (1 + t * A1))

    # 计算整个表达式
    result = term1 + term2 - term3 + term4

    # 比较结果是否等于π（考虑浮点数精度）
    return math.isclose(result, math.pi, rel_tol=1e-9, abs_tol=1e-9)


# 给定的参数值
s = 1.0
t = 1.0
A0 = 0.3090
A1 = 0.0000
B0 = 0.2239
B1 = -0.2239
E00 = 0.7246
E01 = -0.7246
E10 = 0.6554
E11 = 0.6554


# 检查方程是否成立
is_equal = check_equation(s, t, A0, B0, A1, B1, E00, E01, E10, E11)

# 打印结果
print(f"计算结果是否等于π: {is_equal}")

# 如果需要，可以打印实际计算结果和π的差值
if not is_equal:
    term1 = math.asin((E00 + s * B0) / (1 + s * A0))
    term2 = math.asin((E10 + t * B0) / (1 + t * A1))
    term3 = math.asin((E01 + s * B1) / (1 + s * A0))
    term4 = math.asin((E11 + t * B1) / (1 + t * A1))
    result = term1 + term2 - term3 + term4
    print(f"计算结果: {result}")
    print(f"π的值: {math.pi}")
    print(f"差值: {abs(result - math.pi)}")