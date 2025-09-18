import numpy as np
from scipy.optimize import minimize
import math

# 给定的参数值
theta = np.pi / 6
a0 = 0
a1 = 1.42744876
b0 = 0.71372438
b1 = 2.28452071

# 定义点P2
P2 = np.array([0.37796447, 0.37796447, 0.50000000, 0.00000000,
               0.75592895, -0.56694671, 0.75592895, 0.56694671])


# 计算基向量v1的8个分量（已完整实现，此处省略）
def compute_v1(theta, a0, a1, b0, b1):
    # 保持与之前相同的完整实现
    v1 = np.zeros(8)

    # β_1
    numerator = -(2 * np.sin(theta) ** 2 - 1) * np.sin(b0 - b1) * np.cos(a0)
    denominator = 2 * np.sin(a0) * np.sin(theta) * np.cos(b0) * np.cos(theta) - np.sin(b0) * np.cos(a0)
    v1[0] = numerator / denominator if denominator != 0 else 0

    # β_2
    term1 = -np.sin(a0) ** 2 * np.sin(b1) / 2
    term2 = -2 * np.sin(a0) ** 2 * np.sin(theta) ** 4 * np.cos(b1) / np.tan(b0)
    term3 = 2 * np.sin(a0) ** 2 * np.sin(theta) ** 2 * np.cos(b1) / np.tan(b0)
    term4 = np.sin(a0) * np.sin(theta) ** 3 * np.sin(b0 + b1) * np.cos(a0) / (np.sin(b0) * np.cos(theta))
    term5 = -np.sin(a0) * np.sin(b0 + b1) * np.cos(a0) * np.tan(theta) / np.sin(b0)
    term6 = 2 * np.sin(b1) * np.sin(theta) ** 4
    term7 = -2 * np.sin(b1) * np.sin(theta) ** 2
    term8 = np.sin(b1)
    numerator = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8) * np.sin(b0 - b1)

    denom_term1 = -4 * np.sin(a0) * np.sin(theta) ** 3 * np.cos(b0) * np.cos(theta)
    denom_term2 = 2 * np.sin(a0) * np.sin(theta) * np.cos(b0) * np.cos(theta)
    denom_term3 = 2 * np.sin(b0) * np.sin(theta) ** 2 * np.cos(a0)
    denom_term4 = -np.sin(b0) * np.cos(a0)
    denominator = (denom_term1 + denom_term2 + denom_term3 + denom_term4) * np.sin(a1 - b1)
    v1[1] = numerator / denominator if denominator != 0 else 0

    # β_3
    numerator = 2 * np.sin(a0) * np.sin(theta) * np.cos(b1) * np.cos(theta) - np.sin(b1) * np.cos(a0)
    denominator = (1 - 2 * np.cos(theta) ** 2) * np.sin(b0)
    v1[2] = numerator / denominator if denominator != 0 else 0

    # β_4
    term1 = -2 * np.sin(a0) ** 2 * np.sin(a1) * np.sin(theta) ** 4 * np.cos(b1) / np.sin(b0)
    term2 = 2 * np.sin(a0) ** 2 * np.sin(a1) * np.sin(theta) ** 2 * np.cos(b1) / np.sin(b0)
    term3 = -np.sin(a0) ** 2 * np.sin(b1) * np.sin(a1 - b0) / 2
    term4 = 2 * np.sin(a0) ** 2 * np.sin(theta) ** 4 * np.cos(b1) * np.cos(a1 - b0)
    term5 = -2 * np.sin(a0) ** 2 * np.sin(theta) ** 2 * np.cos(b1) * np.cos(a1 - b0)
    term6 = np.sin(a0) * np.sin(a1) * np.sin(b1) * np.sin(theta) ** 3 * np.cos(a0) / (np.sin(b0) * np.cos(theta))
    term7 = -np.sin(a0) * np.sin(a1) * np.sin(b1) * np.cos(a0) * np.tan(theta) / np.sin(b0)
    term8 = -np.sin(a0) * np.sin(theta) ** 3 * np.sin(-a1 + b0 + b1) * np.cos(a0) / np.cos(theta)
    term9 = np.sin(a0) * np.sin(-a1 + b0 + b1) * np.cos(a0) * np.tan(theta)
    term10 = np.sin(a1) * np.sin(b1) * np.cos(b0)
    term11 = -2 * np.sin(a1) * np.sin(theta) ** 4 * np.sin(b0 - b1)
    term12 = 2 * np.sin(a1) * np.sin(theta) ** 2 * np.sin(b0 - b1)
    term13 = -np.sin(b0) * np.sin(a1 + b1) / 2
    numerator = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12 + term13

    denom_term1 = -4 * np.sin(a0) * np.sin(theta) ** 3 * np.cos(b0) * np.cos(theta)
    denom_term2 = 2 * np.sin(a0) * np.sin(theta) * np.cos(b0) * np.cos(theta)
    denom_term3 = 2 * np.sin(b0) * np.sin(theta) ** 2 * np.cos(a0)
    denom_term4 = -np.sin(b0) * np.cos(a0)
    denominator = (denom_term1 + denom_term2 + denom_term3 + denom_term4) * np.sin(a1 - b1)
    v1[3] = numerator / denominator if denominator != 0 else 0

    # β_5
    numerator = 2 * np.sin(a0) * np.sin(theta) * np.cos(b1) * np.cos(theta) - np.sin(b1) * np.cos(a0)
    denominator = -2 * np.sin(a0) * np.sin(theta) * np.cos(b0) * np.cos(theta) + np.sin(b0) * np.cos(a0)
    v1[4] = numerator / denominator if denominator != 0 else 0

    # β_6 to β_8
    v1[5] = 1
    v1[6] = 0
    v1[7] = 0

    return v1


# 计算基向量v2的8个分量（完整实现）
def compute_v2(theta, a0, a1, b0, b1):
    v2 = np.zeros(8)

    # β_1（已完整实现）
    term1 = -2 * np.sin(a0) * np.sin(a1) * np.sin(theta) * np.cos(b0) ** 2 * np.cos(theta)
    term2 = np.sin(a0) * np.sin(b0) * np.cos(a1) * np.cos(b0)
    term3 = 4 * np.sin(a1) * np.sin(b0) * np.sin(theta) ** 2 * np.cos(a0) * np.cos(b0) * np.cos(theta) ** 2
    term4 = -2 * np.sin(b0) ** 2 * np.sin(theta) * np.cos(a0) * np.cos(a1) * np.cos(theta)
    numerator = term1 + term2 + term3 + term4

    denominator = (-2 * np.sin(a0) * np.sin(theta) * np.cos(b0) * np.cos(theta) + np.sin(b0) * np.cos(a0)) * np.sin(
        a0) * np.cos(2 * theta)
    v2[0] = numerator / denominator if denominator != 0 else 0

    # β_2（完整实现）
    term1 = -8 * np.sin(a0) ** 2 * np.sin(b0) * np.sin(theta) ** 5 * np.cos(b0) * np.cos(theta) * np.cos(a1 - b1)
    term2 = 8 * np.sin(a0) ** 2 * np.sin(b0) * np.sin(theta) ** 3 * np.cos(b0) * np.cos(theta) * np.cos(a1 - b1)
    term3 = 8 * np.sin(a0) * np.sin(a1) * np.sin(b0) ** 2 * np.sin(b1) * np.sin(theta) ** 4 * np.cos(a0)
    term4 = -8 * np.sin(a0) * np.sin(a1) * np.sin(b0) ** 2 * np.sin(b1) * np.sin(theta) ** 2 * np.cos(a0)
    term5 = 4 * np.sin(a0) * np.sin(a1) * np.sin(theta) ** 4 * np.sin(a0 - b1)
    term6 = -4 * np.sin(a0) * np.sin(a1) * np.sin(theta) ** 2 * np.sin(a0 - b1)
    term7 = -np.sin(a0) * np.sin(b0) ** 2 * np.sin(b1) * np.sin(a0 - a1)
    term8 = 4 * np.sin(a0) * np.sin(b0) ** 2 * np.sin(theta) ** 4 * np.cos(b1) * np.cos(a0 + a1)
    term9 = -4 * np.sin(a0) * np.sin(b0) ** 2 * np.sin(theta) ** 2 * np.cos(b1) * np.cos(a0 + a1)
    term10 = 2 * np.sin(a0) * np.sin(b0) * np.sin(theta) * np.sin(a1 - b1) * np.cos(a0) * np.cos(b0) * np.cos(theta)
    term11 = 8 * np.sin(a1) * np.sin(b0) * np.sin(b1) * np.sin(theta) ** 5 * np.cos(b0) * np.cos(theta)
    term12 = -8 * np.sin(a1) * np.sin(b0) * np.sin(b1) * np.sin(theta) ** 3 * np.cos(b0) * np.cos(theta)
    term13 = -4 * np.sin(b0) ** 2 * np.sin(b1) * np.sin(theta) ** 4 * np.cos(a1)
    term14 = 4 * np.sin(b0) ** 2 * np.sin(b1) * np.sin(theta) ** 2 * np.cos(a1)
    numerator = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12 + term13 + term14

    denominator = (2 * np.sin(a0) * np.sin(theta) * np.cos(b0) * np.cos(theta) - np.sin(b0) * np.cos(a0)) * np.sin(
        a0) * np.sin(2 * theta) * np.sin(a1 - b1) * np.cos(2 * theta)
    v2[1] = numerator / denominator if denominator != 0 else 0

    # β_3（已完整实现）
    v2[2] = 0

    # β_4（完整实现）
    term1 = 8 * np.sin(a0) * np.sin(a1) * np.sin(b0) ** 2 * np.sin(theta) ** 4
    term2 = -8 * np.sin(a0) * np.sin(a1) * np.sin(b0) ** 2 * np.sin(theta) ** 2
    term3 = np.sin(a0) * np.sin(a1) * np.sin(b0) ** 2
    term4 = -4 * np.sin(a0) * np.sin(a1) * np.sin(theta) ** 4
    term5 = 4 * np.sin(a0) * np.sin(a1) * np.sin(theta) ** 2
    term6 = -4 * np.sin(b0) ** 2 * np.sin(theta) ** 4 * np.cos(a0) * np.cos(a1)
    term7 = 4 * np.sin(b0) ** 2 * np.sin(theta) ** 2 * np.cos(a0) * np.cos(a1)
    term8 = 8 * np.sin(b0) * np.sin(theta) ** 5 * np.sin(a0 + a1) * np.cos(b0) * np.cos(theta)
    term9 = -8 * np.sin(b0) * np.sin(theta) ** 3 * np.sin(a0 + a1) * np.cos(b0) * np.cos(theta)
    numerator = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9) * np.sin(a0 - a1)

    denominator = (2 * np.sin(a0) * np.sin(theta) * np.cos(b0) * np.cos(theta) - np.sin(b0) * np.cos(a0)) * np.sin(
        a0) * np.sin(2 * theta) * np.sin(a1 - b1) * np.cos(2 * theta)
    v2[3] = numerator / denominator if denominator != 0 else 0

    # β_5（已完整实现）
    numerator = 2 * np.sin(a1) * np.sin(theta) * np.cos(b0) * np.cos(theta) - np.sin(b0) * np.cos(a1)
    denominator = -2 * np.sin(a0) * np.sin(theta) * np.cos(b0) * np.cos(theta) + np.sin(b0) * np.cos(a0)
    v2[4] = numerator / denominator if denominator != 0 else 0

    # β_6 to β_8（已完整实现）
    v2[5] = 0
    v2[6] = 1
    v2[7] = 0

    return v2


# 计算基向量v3的8个分量（完整实现）
def compute_v3(theta, a0, a1, b0, b1):
    v3 = np.zeros(8)

    # β_1（完整实现）
    term1 = -2 * np.sin(a0) * np.sin(a1) * np.sin(theta) * np.cos(b0) * np.cos(b1) * np.cos(theta)
    term2 = np.sin(a0) * np.sin(b1) * np.cos(a1) * np.cos(b0)
    term3 = 4 * np.sin(a1) * np.sin(b0) * np.sin(theta) ** 2 * np.cos(a0) * np.cos(b1) * np.cos(theta) ** 2
    term4 = -2 * np.sin(b0) * np.sin(b1) * np.sin(theta) * np.cos(a0) * np.cos(a1) * np.cos(theta)
    numerator = term1 + term2 + term3 + term4

    denominator = (-2 * np.sin(a0) * np.sin(theta) * np.cos(b0) * np.cos(theta) + np.sin(b0) * np.cos(a0)) * np.sin(
        a0) * np.cos(2 * theta)
    v3[0] = numerator / denominator if denominator != 0 else 0

    # β_2（完整实现）
    term1 = -4 * np.sin(a0) * np.sin(a1) * np.sin(theta) ** 4 * np.cos(b0)
    term2 = 4 * np.sin(a0) * np.sin(a1) * np.sin(theta) ** 2 * np.cos(b0)
    term3 = 4 * np.sin(a0) * np.sin(a1) * np.sin(b1) * np.sin(theta) ** 4 * np.cos(b1) / np.sin(b0)
    term4 = -4 * np.sin(a0) * np.sin(a1) * np.sin(b1) * np.sin(theta) ** 2 * np.cos(b1) / np.sin(b0)
    term5 = -2 * np.sin(a0) * np.sin(b1) ** 2 * np.sin(theta) ** 3 * np.sin(a1 - b0) / np.cos(theta)
    term6 = -4 * np.sin(a0) * np.sin(b1) * np.sin(theta) ** 4 * np.cos(b1) * np.cos(a1 - b0)
    term7 = 4 * np.sin(a0) * np.sin(b1) * np.sin(theta) ** 2 * np.cos(b1) * np.cos(a1 - b0)
    term8 = -2 * np.sin(a0) * np.sin(b1) * np.cos(b0) * np.cos(a1 + b1) * np.tan(theta)
    term9 = -2 * np.sin(a0) * np.sin(b1) ** 2 * np.sin(theta) ** 3 * np.cos(a1) / (np.sin(b0) * np.cos(theta))
    term10 = 2 * np.sin(a0) * np.sin(b1) ** 2 * np.cos(a1) * np.tan(theta) / np.sin(b0)
    term11 = np.sin(a1) * np.sin(b0) * np.cos(a0)
    term12 = -np.sin(a1) * np.sin(b1) ** 2 * np.sin(a0 + b0)
    term13 = -4 * np.sin(a1) * np.sin(b1) * np.sin(theta) ** 3 * np.cos(a0) * np.cos(b0) * np.cos(b1) / np.cos(theta)
    term14 = 4 * np.sin(a1) * np.sin(b1) * np.cos(a0) * np.cos(b0) * np.cos(b1) * np.tan(theta)
    term15 = 2 * np.sin(a1) * np.sin(theta) ** 3 * np.sin(a0 + b0) / np.cos(theta)
    term16 = -2 * np.sin(a1) * np.sin(a0 + b0) * np.tan(theta)
    term17 = -np.sin(b0) * np.sin(b1) ** 2 * np.sin(a0 - a1) / (2 * np.sin(theta) * np.cos(theta))
    term18 = -2 * np.sin(b0) * np.sin(b1) * np.cos(a1) * np.cos(a0 - b1) * np.tan(theta)
    term19 = np.sin(b0) * np.sin(b1) * np.cos(b1) * np.cos(a0 - a1)
    term20 = -2 * np.sin(b1) ** 2 * np.cos(a0) * np.cos(a1) * np.cos(b0)
    term21 = 2 * np.sin(b1) * np.sin(theta) ** 3 * np.sin(a0 + b0) * np.cos(a1) * np.cos(b1) / np.cos(theta)
    term22 = 4 * np.sin(a1) * np.sin(b0) * np.sin(b1) * np.sin(theta) ** 4 * np.cos(b1) / np.sin(a0)
    term23 = -4 * np.sin(a1) * np.sin(b0) * np.sin(b1) * np.sin(theta) ** 2 * np.cos(b1) / np.sin(a0)
    term24 = -2 * np.sin(b0) * np.sin(b1) ** 2 * np.sin(theta) ** 3 * np.cos(a1) / (np.sin(a0) * np.cos(theta))
    term25 = 2 * np.sin(b0) * np.sin(b1) ** 2 * np.cos(a1) * np.tan(theta) / np.sin(a0)

    numerator = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12 + term13 + term14 + term15 + term16 + term17 + term18 + term19 + term20 + term21 + term22 + term23 + term24 + term25

    denom_term1 = -4 * np.sin(a0) * np.sin(theta) ** 3 * np.cos(b0) * np.cos(theta)
    denom_term2 = 2 * np.sin(a0) * np.sin(theta) * np.cos(b0) * np.cos(theta)
    denom_term3 = 2 * np.sin(b0) * np.sin(theta) ** 2 * np.cos(a0)
    denom_term4 = -np.sin(b0) * np.cos(a0)
    denominator = (denom_term1 + denom_term2 + denom_term3 + denom_term4) * np.sin(a1 - b1)
    v3[1] = numerator / denominator if denominator != 0 else 0

    # β_3（已完整实现）
    numerator = 2 * np.sin(a1) * np.sin(theta) * np.cos(b1) * np.cos(theta) - np.sin(b1) * np.cos(a1)
    denominator = (1 - 2 * np.cos(theta) ** 2) * np.sin(b0)
    v3[2] = numerator / denominator if denominator != 0 else 0

    # β_4（完整实现）
    term1 = -4 * np.sin(a0) * np.sin(a1) ** 2 * np.sin(theta) ** 4 * np.cos(b1) / np.sin(b0)
    term2 = 4 * np.sin(a0) * np.sin(a1) ** 2 * np.sin(theta) ** 2 * np.cos(b1) / np.sin(b0)
    term3 = 2 * np.sin(a0) * np.sin(a1) * np.sin(b1) * np.sin(theta) ** 3 * np.sin(a1 - b0) / np.cos(theta)
    term4 = 4 * np.sin(a0) * np.sin(a1) * np.sin(theta) ** 4 * np.cos(b1) * np.cos(a1 - b0)
    term5 = -4 * np.sin(a0) * np.sin(a1) * np.sin(theta) ** 2 * np.cos(b1) * np.cos(a1 - b0)
    term6 = 2 * np.sin(a0) * np.sin(a1) * np.cos(b0) * np.cos(a1 + b1) * np.tan(theta)
    term7 = 2 * np.sin(a0) * np.sin(a1) * np.sin(b1) * np.sin(theta) ** 3 * np.cos(a1) / (np.sin(b0) * np.cos(theta))
    term8 = -2 * np.sin(a0) * np.sin(a1) * np.sin(b1) * np.cos(a1) * np.tan(theta) / np.sin(b0)
    term9 = 4 * np.sin(a0) * np.sin(b1) * np.sin(theta) ** 4 * np.cos(b0)
    term10 = -4 * np.sin(a0) * np.sin(b1) * np.sin(theta) ** 2 * np.cos(b0)
    term11 = np.sin(a1) ** 2 * np.sin(b1) * np.sin(a0 + b0)
    term12 = 4 * np.sin(a1) ** 2 * np.sin(theta) ** 3 * np.cos(a0) * np.cos(b0) * np.cos(b1) / np.cos(theta)
    term13 = -4 * np.sin(a1) ** 2 * np.cos(a0) * np.cos(b0) * np.cos(b1) * np.tan(theta)
    term14 = np.sin(a1) * np.sin(b0) * np.sin(b1) * np.sin(a0 - a1) / (2 * np.sin(theta) * np.cos(theta))
    term15 = 2 * np.sin(a1) * np.sin(b0) * np.cos(a1) * np.cos(a0 - b1) * np.tan(theta)
    term16 = -np.sin(a1) * np.sin(b0) * np.cos(b1) * np.cos(a0 - a1)
    term17 = 2 * np.sin(a1) * np.sin(b1) * np.cos(a0) * np.cos(a1) * np.cos(b0)
    term18 = -2 * np.sin(a1) * np.sin(theta) ** 3 * np.sin(a0 + b0) * np.cos(a1) * np.cos(b1) / np.cos(theta)
    term19 = -np.sin(b0) * np.sin(b1) * np.cos(a0)
    term20 = -2 * np.sin(b1) * np.sin(theta) ** 3 * np.sin(a0 + b0) / np.cos(theta)
    term21 = 2 * np.sin(b1) * np.sin(a0 + b0) * np.tan(theta)
    term22 = -4 * np.sin(a1) ** 2 * np.sin(b0) * np.sin(theta) ** 4 * np.cos(b1) / np.sin(a0)
    term23 = 4 * np.sin(a1) ** 2 * np.sin(b0) * np.sin(theta) ** 2 * np.cos(b1) / np.sin(a0)
    term24 = 2 * np.sin(a1) * np.sin(b0) * np.sin(b1) * np.sin(theta) ** 3 * np.cos(a1) / (np.sin(a0) * np.cos(theta))
    term25 = -2 * np.sin(a1) * np.sin(b0) * np.sin(b1) * np.cos(a1) * np.tan(theta) / np.sin(a0)

    numerator = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12 + term13 + term14 + term15 + term16 + term17 + term18 + term19 + term20 + term21 + term22 + term23 + term24 + term25

    denom_term1 = -4 * np.sin(a0) * np.sin(theta) ** 3 * np.cos(b0) * np.cos(theta)
    denom_term2 = 2 * np.sin(a0) * np.sin(theta) * np.cos(b0) * np.cos(theta)
    denom_term3 = 2 * np.sin(b0) * np.sin(theta) ** 2 * np.cos(a0)
    denom_term4 = -np.sin(b0) * np.cos(a0)
    denominator = (denom_term1 + denom_term2 + denom_term3 + denom_term4) * np.sin(a1 - b1)
    v3[3] = numerator / denominator if denominator != 0 else 0

    # β_5（已完整实现）
    numerator = 2 * np.sin(a1) * np.sin(theta) * np.cos(b1) * np.cos(theta) - np.sin(b1) * np.cos(a1)
    denominator = -2 * np.sin(a0) * np.sin(theta) * np.cos(b0) * np.cos(theta) + np.sin(b0) * np.cos(a0)
    v3[4] = numerator / denominator if denominator != 0 else 0

    # β_6 to β_8（已完整实现）
    v3[5] = 0
    v3[6] = 0
    v3[7] = 1

    return v3


# 计算三个基向量
v1 = compute_v1(theta, a0, a1, b0, b1)
v2 = compute_v2(theta, a0, a1, b0, b1)
v3 = compute_v3(theta, a0, a1, b0, b1)

print("基向量v1:", v1)
print("基向量v2:", v2)
print("基向量v3:", v3)


# 定义目标函数
def objective(x):
    u, v, w = x
    beta = u * v1 + v * v2 + w * v3
    return -np.dot(beta, P2)  # 负号因为我们要最大化点积


# 初始猜测值
x0 = [1, 1, 1]

# 优化求解
solution = minimize(objective, x0, method='SLSQP')
u_opt, v_opt, w_opt = solution.x

# 计算最优解对应的贝尔表达式值
beta_opt = u_opt * v1 + v_opt * v2 + w_opt * v3
max_value = np.dot(beta_opt, P2)

print("\n优化结果:")
print(f"u = {u_opt}")
print(f"v = {v_opt}")
print(f"w = {w_opt}")
print(f"贝尔表达式在P2点的最大值: {max_value}")
