def calculate_general_bell_max(coefficients):
    """
    计算一般贝尔不等式的经典最大值
    coefficients: (a0, a1, b0, b1, e00, e01, e10, e11)
    """
    a0, a1, b0, b1, e00_coef, e01_coef, e10_coef, e11_coef = coefficients

    max_value = -float('inf')
    best_config = None

    for A0 in [-1, 1]:
        for A1 in [-1, 1]:
            for B0 in [-1, 1]:
                for B1 in [-1, 1]:
                    # 计算E_ij = A_i × B_j
                    E00_val = A0 * B0
                    E01_val = A0 * B1
                    E10_val = A1 * B0
                    E11_val = A1 * B1

                    # 计算贝尔值
                    value = (a0 * A0 + a1 * A1 + b0 * B0 + b1 * B1 +
                             e00_coef * E00_val - e01_coef * E01_val +
                             e10_coef * E10_val + e11_coef * E11_val)

                    if value > max_value:
                        max_value = value
                        best_config = (A0, A1, B0, B1, E00_val, E01_val, E10_val, E11_val)

    return max_value, best_config


# 使用您提供的系数
coefficients = (-1.9352216957983341, 3.5276685151424334,
                -1.9352216957983341, 0,
                1.7638342575712167, -0.7962233898452387,
                1.7638342575712167, 0.7962233898452387)

max_value, config = calculate_general_bell_max(coefficients)
print(f"一般贝尔不等式的经典最大值: {max_value}")