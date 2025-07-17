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


def check_angle_relations(a0, a1, b0, b1, theta, s, t):
    """检查角度关系是否满足 0 <= [a0s] <= b0 <= [a1t] <= b1 < pi"""
    # 计算 [a0s] 和 [a1t]
    a0s = calculate_angle_mod(a0, theta, s)
    a1t = calculate_angle_mod(a1, theta, t)

    # 打印输入参数和计算结果
    print("输入参数和计算结果:")
    print(f"a0 = {a0:.6f} rad, a1 = {a1:.6f} rad")
    print(f"b0 = {b0:.6f} rad, b1 = {b1:.6f} rad")
    print(f"theta = {theta:.6f} rad, s = {s}, t = {t}")
    print(f"[a0s] = {a0s:.6f} rad, [a1t] = {a1t:.6f} rad")

    # 检查条件
    conditions = [
        (0 <= a0s, f"0 <= [a0s] ({a0s:.6f})"),
        (a0s <= b0, f"[a0s] ({a0s:.6f}) <= b0 ({b0:.6f})"),
        (b0 <= a1t, f"b0 ({b0:.6f}) <= [a1t] ({a1t:.6f})"),
        (a1t <= b1, f"[a1t] ({a1t:.6f}) <= b1 ({b1:.6f})"),
        (b1 < math.pi, f"b1 ({b1:.6f}) < pi ({math.pi:.6f})")
    ]

    print("\n条件检查:")
    all_conditions_met = True
    for condition, desc in conditions:
        status = "满足" if condition else "不满足"
        print(f"{desc}: {status}")
        if not condition:
            all_conditions_met = False

    print("\n结论: 所有角度关系条件" + ("满足!" if all_conditions_met else "不满足!"))
    return all_conditions_met


# 在这里直接修改您需要的参数值
theta = math.pi / 4  # 可以修改
a0 = 0.0  # 可以修改
a1 = math.pi / 2  # 可以修改
b0 = np.arctan(np.sin(2 * theta))  # 可以修改
b1 = np.arctan(np.sin(2 * theta))  # 可以修改
s = 1  # 可以修改
t = 1  # 可以修改

# 检查角度关系
check_angle_relations(a0, a1, b0, b1, theta, s, t)