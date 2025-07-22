import math
import itertools

# 定义常量
A0 = 0
A1 = 0
B0 = 0
B1 = 0  # chsh是0，tilted是arctan(sin2theta)
E00 = math.sqrt(2) / 2
E01 = math.sqrt(2) / 2
E10 = math.sqrt(2) / 2
E11 = -math.sqrt(2) / 2

# 定义s和t的可能取值
s_values = [1, -1]
t_values = [1, -1]


# 计算[sA0B0], [tA1B0], [sA0B1], [tA1B1]的函数
def calculate_terms(s, t):
    sA0B0 = (E00 + s * B0) / (1 + s * A0)
    tA1B0 = (E10 + t * B0) / (1 + t * A1)
    sA0B1 = (E01 + s * B1) / (1 + s * A0)
    tA1B1 = (E11 + t * B1) / (1 + t * A1)
    return sA0B0, tA1B0, sA0B1, tA1B1


# 计算公式左边的值
def calculate_formula(s, t):
    sA0B0, tA1B0, sA0B1, tA1B1 = calculate_terms(s, t)

    # 确保所有值都在arcsin的定义域内[-1, 1]
    terms = [sA0B0, tA1B0, -sA0B1, tA1B1]
    for term in terms:
        if abs(term) > 1:
            return None  # 超出定义域

    value = math.asin(sA0B0) + math.asin(tA1B0) - math.asin(sA0B1) + math.asin(tA1B1)
    return value


# 验证所有组合
for s, t in itertools.product(s_values, t_values):
    print(f"\n验证组合: s={s}, t={t}")

    # 计算中间项
    sA0B0, tA1B0, sA0B1, tA1B1 = calculate_terms(s, t)
    print(f"[sA0B0] = {sA0B0}")
    print(f"[tA1B0] = {tA1B0}")
    print(f"[sA0B1] = {sA0B1}")
    print(f"[tA1B1] = {tA1B1}")

    # 检查定义域
    terms = [sA0B0, tA1B0, sA0B1, tA1B1]
    valid = True
    for term in terms:
        if abs(term) > 1:
            valid = False
            break

    if not valid:
        print("警告: 某些值超出arcsin的定义域[-1, 1]，无法计算")
        continue

    # 计算并比较
    calculated = calculate_formula(s, t)
    if calculated is not None:
        print(f"计算值: {calculated}")
        print(f"pi的值: {math.pi}")
        print(f"是否等于pi: {math.isclose(calculated, math.pi, rel_tol=1e-9)}")
    else:
        print("无法计算，因为某些值超出arcsin的定义域")