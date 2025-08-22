import numpy as np


def compute_theta_min(a0, a1, b0, b1):
    """
    计算 theta_min = max{ theta <= pi/4 | a0minus=b0 或 b0=a1plus 或 a1minus=b1 }
    返回满足条件的最大 theta（如果存在），否则返回 None。
    """
    solutions = []  # 存储所有有效解（theta <= pi/4）

    # 方程 1: a0minus = b0
    try:
        numerator = np.tan(a0 / 2)
        denominator = np.tan(b0 / 2)
        if denominator != 0:
            theta = np.arctan(numerator / denominator)
            if theta <= np.pi / 4:
                solutions.append(("a0minus = b0", theta))
            else:
                print(f"方程 a0minus = b0 有解 theta = {theta:.4f} > pi/4")
        else:
            print("方程 a0minus = b0 无解（分母为零）")
    except Exception as e:
        print(f"方程 a0minus = b0 计算错误: {e}")

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
theta_origin = np.pi / 6

sin2theta = np.sin(2 * theta_origin)

# Calculate mu
mu = np.arctan(sin2theta)

a0 = 0
a1 = 2 * mu
b0 = mu
b1 = np.pi / 2 + mu

print("=== 输入参数 ===")
print(f"a0 = {a0:.4f}, a1 = {a1:.4f}, b0 = {b0:.4f}, b1 = {b1:.4f}\n")

theta_min = compute_theta_min(a0, a1, b0, b1)