import numpy as np


def compute_theta_min(a0, a1, b0, b1, theta_test=np.pi / 6):
    """
    计算 theta_min = max{ theta <= pi/4 | a0plus=b0 或 b0=a1plus 或 a1minus=b1 }
    返回满足条件的最大 theta（如果存在），否则返回 None。
    """
    # First calculate all the required values
    tan_theta = np.tan(theta_test)

    a0plus = 2 * np.arctan(np.tan(a0 / 2) * (tan_theta) ** 1)
    a0minus = 2 * np.arctan(np.tan(a0 / 2) * (tan_theta) ** -1)
    a1plus = 2 * np.arctan(np.tan(a1 / 2) * (tan_theta) ** 1)
    a1minus = 2 * np.arctan(np.tan(a1 / 2) * (tan_theta) ** -1)

    print("=== 计算中间值 ===")
    print(f"a0plus = {a0plus:.4f}")
    print(f"a0minus = {a0minus:.4f}")
    print(f"a1plus = {a1plus:.4f}")
    print(f"a1minus = {a1minus:.4f}")
    print(f"b0 = {b0:.4f}")
    print(f"b1 = {b1:.4f}\n")

    solutions = []  # 存储所有有效解（theta <= pi/4）

    # 方程 1: a0plus = b0
    try:
        numerator = np.tan(b0 / 2)
        denominator = np.tan(a0 / 2)
        if denominator != 0:
            theta = np.arctan(numerator / denominator)
            if theta <= np.pi / 4:
                solutions.append(("a0plus = b0", theta))
            else:
                print(f"方程 a0plus = b0 有解 theta = {theta:.4f} > pi/4")
        else:
            print("方程 a0plus = b0 无解（分母为零）")
    except Exception as e:
        print(f"方程 a0plus = b0 计算错误: {e}")

    # 方程 2: b0 = a1plus
    try:
        numerator = np.tan(b0 / 2)
        denominator = np.tan(a1 / 2)
        if denominator != 0:
            theta = np.arctan(numerator / denominator)
            if theta <= np.pi / 4:
                solutions.append(("b0 = a1plus", theta))
            else:
                print(f"方程 b0 = a1plus 有解 theta = {theta:.4f} > pi/4")
        else:
            print("方程 b0 = a1plus 无解（分母为零）")
    except Exception as e:
        print(f"方程 b0 = a1plus 计算错误: {e}")

    # 方程 3: a1minus = b1
    try:
        numerator = np.tan(a1 / 2)
        denominator = np.tan(b1 / 2)
        if denominator != 0:
            theta = np.arctan(numerator / denominator)
            if theta <= np.pi / 4:
                solutions.append(("a1minus = b1", theta))
            else:
                print(f"方程 a1minus = b1 有解 theta = {theta:.4f} > pi/4")
        else:
            print("方程 a1minus = b1 无解（分母为零）")
    except Exception as e:
        print(f"方程 a1minus = b1 计算错误: {e}")

    # 输出所有有效解
    print("\n满足条件的解：")
    if solutions:
        for eq, theta in solutions:
            print(f"- {eq}: theta = {theta:.4f} (≈{np.degrees(theta):.2f}°)")
        theta_min = max(theta for _, theta in solutions)
        print(f"\n最大值 theta_min = {theta_min:.4f} (≈{np.degrees(theta_min):.2f}°)")
        return theta_min
    else:
        print("无满足条件的解")
        return None


# 示例测试
theta = np.pi / 6
a0 = -np.arctan(np.sin(2 * theta))
a1 = np.arctan(np.sin(2 * theta))
b0 = 0.0
b1 = np.pi / 2

print("=== 输入参数 ===")
print(f"a0 = {a0:.4f}, a1 = {a1:.4f}, b0 = {b0:.4f}, b1 = {b1:.4f}")
print(f"测试 theta = {theta:.4f} (≈{np.degrees(theta):.2f}°)\n")

theta_min = compute_theta_min(a0, a1, b0, b1, theta)