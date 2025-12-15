import numpy as np


def calculate_specific_theta():
    # 1. 设置 theta 固定值为 pi/6
    theta_rad = np.pi / 6

    # 2. 计算 2theta 的三角函数
    sin_2theta = np.sin(2 * theta_rad)
    cos_2theta = np.cos(2 * theta_rad)

    # 3. 计算 alpha
    # 公式推导: sin(2theta)^2 = (1 - alpha^2/4) / (1 + alpha^2/4)
    # 解得 alpha = 2 * |cos(2theta)| / sqrt(1 + sin(2theta)^2)
    numerator = cos_2theta ** 2
    denominator = 1 + sin_2theta ** 2
    alpha = 2 * np.sqrt(numerator / denominator)

    # 4. 计算 mu
    # 公式: tan(mu) = sin(2theta)
    mu = np.arctan(sin_2theta)

    # 5. 计算 mu 的三角函数
    sin_2mu = np.sin(2 * mu)
    cos_2mu = np.cos(2 * mu)

    # 6. 计算 k1, k2
    k1 = cos_2mu / sin_2mu
    k2 = 1 / sin_2mu

    # 7. 计算 Quantum Bound (8 + 2*alpha^2)^(1/2)
    quantum_bound = np.sqrt(8 + 2 * alpha ** 2)

    # 打印结果
    print("-" * 35)
    print(f"设定参数: Theta = pi/6 (30°)")
    print("-" * 35)
    print(f"theta (rad)      : {theta_rad:.8f}")
    print(f"alpha            : {alpha:.8f}")
    print(f"mu (rad)         : {mu:.8f}")
    print("-" * 35)
    print(f"sin(2theta)      : {sin_2theta:.8f}")
    print(f"cos(2theta)      : {cos_2theta:.8f}")
    print(f"sin(2mu)         : {sin_2mu:.8f}")
    print(f"cos(2mu)         : {cos_2mu:.8f}")
    print("-" * 35)
    print(f"k1               : {k1:.8f}")
    print(f"k2               : {k2:.8f}")
    print("-" * 35)
    # [新增输出]
    print(f"sqrt(8+2*alpha^2): {quantum_bound:.8f}")
    print("-" * 35)


if __name__ == "__main__":
    calculate_specific_theta()