import numpy as np


def calculate_expectation(state, operator_matrix):
    """
    计算期望值 <psi| Op |psi>
    state: 列向量 (4x1)
    operator_matrix: 矩阵 (4x4)
    """
    # 注意：在numpy中，对于实数矩阵，共轭转置 .conj().T 等同于转置 .T
    # 但为了通用性保留共轭写法
    return np.real(state.conj().T @ operator_matrix @ state)[0, 0]


def main():
    # ==========================================
    # 1. 参数设置与计算 (Requirement 2)
    # ==========================================
    theta = np.pi / 6  # 30度

    # 计算 theta 相关的三角函数
    sin2theta = np.sin(2 * theta)
    cos2theta = np.cos(2 * theta)

    # 计算 alpha
    # 公式: sin(2theta) = sqrt((1 - alpha^2/4) / (1 + alpha^2/4))
    # 反解: alpha = 2 * |cos(2theta)| / sqrt(1 + sin(2theta)^2)
    alpha = 2 * np.abs(cos2theta) / np.sqrt(1 + sin2theta ** 2)

    # 计算 mu
    # 公式: tan(mu) = sin(2theta)
    mu = np.arctan(sin2theta)

    # 打印 alpha 和 mu
    print("=" * 40)
    print("参数计算结果:")
    print(f"Theta (rad) : {theta:.6f}")
    print(f"Alpha       : {alpha:.6f}")
    print(f"Mu (rad)    : {mu:.6f}")
    print("=" * 40)

    # ==========================================
    # 2. 定义量子态与算符
    # ==========================================

    # 基础 Pauli 矩阵
    I2 = np.eye(2)
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])

    # --- 构建量子态 |\overline{\psi}> ---
    # 系数
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_mu_2 = np.cos(mu / 2)
    s_mu_2 = np.sin(mu / 2)

    # |00>, |01>, |10>, |11> 的系数
    coeff_00 = c_theta * c_mu_2
    coeff_01 = -c_theta * s_mu_2
    coeff_10 = s_theta * s_mu_2
    coeff_11 = s_theta * c_mu_2

    # 状态向量 (4x1)
    psi = np.array([[coeff_00], [coeff_01], [coeff_10], [coeff_11]])

    # 归一化检查 (可选，确保物理意义正确)
    norm = np.linalg.norm(psi)
    psi = psi / norm

    # --- 构建测量算符 (4x4 矩阵) ---
    # Alice 算符 (A otimes I)
    A0 = np.kron(Z, I2)
    A1 = np.kron(X, I2)

    # Bob 算符 (I otimes B)
    B0 = np.kron(I2, Z)
    B2 = np.kron(I2, X)

    # B1 = cos(2mu)Z - sin(2mu)X
    B1_local = np.cos(2 * mu) * Z - np.sin(2 * mu) * X
    B1 = np.kron(I2, B1_local)

    # ==========================================
    # 3. 计算期望值 (Requirement 1)
    # ==========================================

    # 为了方便后续公式调用，我们将矩阵存储在字典中，方便组合乘积
    # 注意：矩阵乘法顺序 A @ B

    # 定义需要计算的算符组合列表
    # 格式: (名称, 矩阵乘积表达式)
    vals = {}

    # 基础项
    vals['A0'] = A0
    vals['B0'] = B0
    vals['B2'] = B2

    # 2项乘积
    vals['A0B0'] = A0 @ B0
    vals['A0B1'] = A0 @ B1
    vals['A1B0'] = A1 @ B0
    vals['A1B1'] = A1 @ B1
    vals['A0B2'] = A0 @ B2
    vals['A1B2'] = A1 @ B2

    # 3项及以上乘积 (注意矩阵乘法顺序，左边先作用)
    # A0 A1 A0 B0 -> (A0 @ A1 @ A0) @ B0
    vals['A0A1A0B0'] = A0 @ A1 @ A0 @ B0

    # B0 B2 B0 -> I @ (Z X Z) -> B0 @ B2 @ B0
    vals['B0B2B0'] = B0 @ B2 @ B0

    # A0 B0 B2 B0 -> A0 @ (B0 @ B2 @ B0)
    vals['A0B0B2B0'] = A0 @ B0 @ B2 @ B0

    # A1 B0 B2 B0
    vals['A1B0B2B0'] = A1 @ B0 @ B2 @ B0

    # A0 A1 A0 B2
    vals['A0A1A0B2'] = A0 @ A1 @ A0 @ B2

    # A0 A1 A0 B0 B2 B0
    vals['A0A1A0B0B2B0'] = A0 @ A1 @ A0 @ B0 @ B2 @ B0

    # 4算符 交换项
    vals['A0A1B0B2'] = A0 @ A1 @ B0 @ B2
    vals['A0A1B2B0'] = A0 @ A1 @ B2 @ B0
    vals['A1A0B0B2'] = A1 @ A0 @ B0 @ B2
    vals['A1A0B2B0'] = A1 @ A0 @ B2 @ B0

    # 计算并存储所有期望值
    expectations = {}
    print("\n--- 期望值列表 ---")

    # 按用户请求的顺序打印
    keys_order = [
        'A0', 'B0', 'A0B0', 'A0B1', 'A1B0', 'A1B1',
        'A0A1A0B0',
        'B2', 'B0B2B0', 'A0B2', 'A0B0B2B0',
        'A1B2', 'A1B0B2B0', 'A0A1A0B2', 'A0A1A0B0B2B0',
        'A0A1B0B2', 'A0A1B2B0', 'A1A0B0B2', 'A1A0B2B0'
    ]

    for key in keys_order:
        val = calculate_expectation(psi, vals[key])
        expectations[key] = val
        print(f"<{key}> = {val:.8f}")

    # ==========================================
    # 4. 计算最终公式 (Requirement 3)
    # ==========================================

    # 公式 1: alpha*<A0> + <A0B0> + <A0B1> + <A1B0> - <A1B1>
    # 注意：alphaA0 理解为 alpha * <A0>
    res_formula_1 = (alpha * expectations['A0'] +
                     expectations['A0B0'] +
                     expectations['A0B1'] +
                     expectations['A1B0'] -
                     expectations['A1B1'])

    # 公式 2: F
    # 为了代码整洁，提前提取常用三角函数
    sin_mu = np.sin(mu)
    cos_mu = np.cos(mu)

    # 逐行构建 F
    # F = 1/4 + ...
    F = 0.25

    # + 1/4 cos(2theta) <A0>
    F += 0.25 * cos2theta * expectations['A0']

    # + 1/4 cos(mu) cos(2theta) <B0>
    F += 0.25 * cos_mu * cos2theta * expectations['B0']

    # + 1/4 cos(mu) <A0B0>
    F += 0.25 * cos_mu * expectations['A0B0']

    # + sin(2theta)sin(mu)/8 * (<A1B0> - <A0A1A0B0>)
    term_group_2 = (sin2theta * sin_mu / 8.0) * (expectations['A1B0'] - expectations['A0A1A0B0'])
    F += term_group_2

    # - sin(mu)cos(2theta)/8 * (<B2> - <B0B2B0>)
    # 注意：原公式是 - coeff * <B2> + coeff * <B0B2B0> = - coeff * (<B2> - <B0B2B0>)
    term_group_3 = (sin_mu * cos2theta / 8.0) * (-expectations['B2'] + expectations['B0B2B0'])
    F += term_group_3

    # - sin(mu)/8 * (<A0B2> - <A0B0B2B0>)
    term_group_4 = (sin_mu / 8.0) * (-expectations['A0B2'] + expectations['A0B0B2B0'])
    F += term_group_4

    # + sin(2theta)cos(mu)/16 * (<A1B2> - <A1B0B2B0> - <A0A1A0B2> + <A0A1A0B0B2B0>)
    term_group_5 = (sin2theta * cos_mu / 16.0) * (
            expectations['A1B2'] - expectations['A1B0B2B0'] -
            expectations['A0A1A0B2'] + expectations['A0A1A0B0B2B0']
    )
    F += term_group_5

    # + sin(2theta)/16 * (<A0A1B0B2> - <A0A1B2B0> - <A1A0B0B2> + <A1A0B2B0>)
    term_group_6 = (sin2theta / 16.0) * (
            expectations['A0A1B0B2'] - expectations['A0A1B2B0'] -
            expectations['A1A0B0B2'] + expectations['A1A0B2B0']
    )
    F += term_group_6

    print("\n" + "=" * 40)
    print("公式计算结果:")
    print("=" * 40)
    print(f"Ineq 1 (alpha*A0 + CHSH_like): {res_formula_1:.8f}")
    print(f"F (Fidelity Witness)         : {F:.8f}")
    print("=" * 40)


if __name__ == "__main__":
    main()