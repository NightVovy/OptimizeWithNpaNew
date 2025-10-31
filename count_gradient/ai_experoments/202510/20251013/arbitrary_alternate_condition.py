import math


def calculate_angle(base_angle, theta, exponent):
    """
    计算 a_s 或 a_t 的值
    公式: 2 * arctan(tan(base_angle/2) * (tan(theta))^exponent)
    """
    if base_angle == math.pi:
        return math.pi

    tan_half_base = math.tan(base_angle / 2)
    tan_theta = math.tan(theta)

    # 处理特殊情况
    if tan_theta == 0 and exponent > 0:
        return 0
    if tan_theta == 0 and exponent < 0:
        return math.pi

    result = 2 * math.atan(tan_half_base * (tan_theta ** exponent))
    return result


def verify_conditions(a0, b0, a1, b1, theta):
    """
    验证所有条件
    """
    # 计算所有可能的组合
    a0_plus = calculate_angle(a0, theta, 1)  # s=1
    a0_minus = calculate_angle(a0, theta, -1)  # s=-1
    a1_plus = calculate_angle(a1, theta, 1)  # t=1
    a1_minus = calculate_angle(a1, theta, -1)  # t=-1

    print(f"计算结果:")
    print(f"a0_plus = {math.degrees(a0_plus):.6f}°")
    print(f"a0_minus = {math.degrees(a0_minus):.6f}°")
    print(f"a1_plus = {math.degrees(a1_plus):.6f}°")
    print(f"a1_minus = {math.degrees(a1_minus):.6f}°")
    print(f"b0 = {math.degrees(b0):.6f}°")
    print(f"b1 = {math.degrees(b1):.6f}°")
    print()

    # 条件1: 0 < a0_plus < a0_minus, a1_plus < a1_minus
    cond1_part1 = 0 < a0_plus < a0_minus
    cond1_part2 = a1_plus < a1_minus
    cond1 = cond1_part1 and cond1_part2

    # 条件2: a0_minus < b0
    cond2 = a0_minus < b0

    # 条件3: b0 < a1_plus
    cond3 = b0 < a1_plus

    # 条件4: a1_minus < b1 < pi
    cond4 = a1_minus < b1 < math.pi

    # 条件5: 0 <= a0 <= b0 <= a1 <= b1 < pi
    cond5 = 0 <= a0 <= b0 <= a1 <= b1 < math.pi

    print("验证结果:")
    print(f"条件1 (0 < a0_plus < a0_minus 且 a1_plus < a1_minus): {cond1}")
    if not cond1:
        print(f"  - 0 < a0_plus < a0_minus: {cond1_part1}")
        print(f"  - a1_plus < a1_minus: {cond1_part2}")

    print(f"条件2 (a0_minus < b0): {cond2}")
    print(f"条件3 (b0 < a1_plus): {cond3}")
    print(f"条件4 (a1_minus < b1 < π): {cond4}")
    print(f"条件5 (0 <= a0 <= b0 <= a1 <= b1 < π): {cond5}")

    all_conditions = cond1 and cond2 and cond3 and cond4 and cond5
    print(f"\n所有条件是否满足: {all_conditions}")

    return all_conditions


# 主程序
if __name__ == "__main__":
    # 将角度转换为弧度
    pi = math.pi

    # 给定的参数
    a0_deg = 0
    b0_deg = 0
    a1_deg = 25 - 5  # 20度
    b1_deg = 95 - 19  # 76度
    theta_deg = 25

    # 转换为弧度
    a0 = math.radians(a0_deg)
    b0 = math.radians(b0_deg)
    a1 = math.radians(a1_deg)
    b1 = math.radians(b1_deg)
    theta = math.radians(theta_deg)

    print(f"输入参数 (角度):")
    print(f"a0 = {a0_deg}°, b0 = {b0_deg}°")
    print(f"a1 = {a1_deg}°, b1 = {b1_deg}°")
    print(f"theta = {theta_deg}°")
    print()

    # 验证条件
    result = verify_conditions(a0, b0, a1, b1, theta)