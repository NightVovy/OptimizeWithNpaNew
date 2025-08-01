import numpy as np


def find_a0(theta):
    # 计算右边
    right_side = np.arctan(np.sin(2 * theta))

    # 解方程求x = tan(a0/2)
    x = np.tan(theta) * np.tan(right_side / 2)

    # 计算a0
    a0 = 2 * np.arctan(x)

    return a0


# 测试theta = pi/6
theta = np.pi / 6
a0 = find_a0(theta)

print(f"当 theta = {theta} (π/6 ≈ {np.pi / 6:.6f}) 时，满足方程的 a0 = {a0:.6f} (≈ {np.degrees(a0):.4f}°)")