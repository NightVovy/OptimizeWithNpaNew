import math
import numpy as np


def calculate_angle_mod(a, theta, power):
    """计算 2arctan(tan(a/2) * tan(theta)^power) mod pi"""
    if math.isclose(a % math.pi, 0):
        return 0.0  # 处理tan(0)和tan(pi)的情况
    tan_half = math.tan(a / 2)
    tan_pow = math.tan(theta) ** power
    angle = 2 * math.atan(tan_half * tan_pow)
    return angle % math.pi


def check_angle_relations(a0, a1, b0, b1, theta):
    """检查角度关系是否满足条件"""
    # 计算 [a0+], [a0-], [a1+], [a1-]
    a0_plus = calculate_angle_mod(a0, theta, 1)
    a0_minus = calculate_angle_mod(a0, theta, -1)
    a1_plus = calculate_angle_mod(a1, theta, 1)
    a1_minus = calculate_angle_mod(a1, theta, -1)

    # 按照要求格式输出各个角度
    print("角度计算结果:")
    print(f"a0+ = {a0_plus:.6f}")
    print(f"a0- = {a0_minus:.6f}")
    print(f"b0 = {b0:.6f}")
    print(f"a1+ = {a1_plus:.6f}")
    print(f"a1- = {a1_minus:.6f}")
    print(f"b1 = {b1:.6f}")

    # 检查条件1: [a0+]<=[a0-]且[a1+]<=[a1-]
    condition1 = (a0_plus <= a0_minus) and (a1_plus <= a1_minus)

    # 检查条件2: 0<=max([a0+],[a0-])<=b0<=min([a1+],[a1-])<=b1<pi
    max_a0 = max(a0_plus, a0_minus)
    min_a1 = min(a1_plus, a1_minus)

    conditions2 = [
        (0 <= max_a0, f"0 <= max([a0+], [a0-]) ({max_a0:.6f})"),
        (max_a0 <= b0, f"max([a0+], [a0-]) ({max_a0:.6f}) <= b0 ({b0:.6f})"),
        (b0 <= min_a1, f"b0 ({b0:.6f}) <= min([a1+], [a1-]) ({min_a1:.6f})"),
        (min_a1 <= b1, f"min([a1+], [a1-]) ({min_a1:.6f}) <= b1 ({b1:.6f})"),
        (b1 < math.pi, f"b1 ({b1:.6f}) < pi ({math.pi:.6f})")
    ]

    print("\n条件1检查: [a0+] <= [a0-] 且 [a1+] <= [a1-]")
    print(f"[a0+] ({a0_plus:.6f}) <= [a0-] ({a0_minus:.6f}): {'满足' if a0_plus <= a0_minus else '不满足'}")
    print(f"[a1+] ({a1_plus:.6f}) <= [a1-] ({a1_minus:.6f}): {'满足' if a1_plus <= a1_minus else '不满足'}")
    print(f"条件1: {'满足' if condition1 else '不满足'}")

    print("\n条件2检查: 0 <= max([a0+], [a0-]) <= b0 <= min([a1+], [a1-]) <= b1 < pi")
    all_conditions2_met = True
    for condition, desc in conditions2:
        status = "满足" if condition else "不满足"
        print(f"{desc}: {status}")
        if not condition:
            all_conditions2_met = False

    print("\n结论:")
    print(f"条件1: {'满足' if condition1 else '不满足'}")
    print(f"条件2: {'满足' if all_conditions2_met else '不满足'}")
    print(f"所有条件: {'满足!' if (condition1 and all_conditions2_met) else '不满足!'}")

    return condition1 and all_conditions2_met


# 在这里直接修改您需要的参数值
theta = math.pi / 6  # 可以修改
a0 = 0.0  # 可以修改
a1 = math.pi / 2  # 可以修改
b0 = np.arctan(np.sin(2 * theta))  # 可以修改
# b1 = - np.arctan(np.sin(2 * theta))  # 可以修改
b1 = np.pi - b0  # 可以修改

# 检查角度关系
check_angle_relations(a0, a1, b0, b1, theta)