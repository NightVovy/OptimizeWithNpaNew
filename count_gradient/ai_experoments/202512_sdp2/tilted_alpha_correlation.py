import numpy as np


def get_tilted_expectations(alpha):
    """
    计算给定 alpha 下 Tilted-CHSH 最大违背的 8 个期望值。
    返回顺序: [A0, A1, B0, B1, E00, E01, E10, E11]
    注意：为了与 linear_independence_check 对接，两体项命名为 E00...
    """
    # 1. 计算参数 theta 和 mu
    # sin(2*theta) = sqrt((1 - alpha^2/4) / (1 + alpha^2/4))
    term_numerator = 1 - (alpha ** 2) / 4.0
    term_denominator = 1 + (alpha ** 2) / 4.0
    sin_2theta = np.sqrt(term_numerator / term_denominator)

    # theta 和 mu
    theta = 0.5 * np.arcsin(sin_2theta)
    mu = np.arctan(sin_2theta)

    # 2. 定义泡利矩阵和单位矩阵
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)

    # 3. 构建单体测量算符
    # Alice
    A0_single = Z
    A1_single = X

    # Bob
    cos_mu = np.cos(mu)
    sin_mu = np.sin(mu)
    B0_single = cos_mu * Z + sin_mu * X
    B1_single = cos_mu * Z - sin_mu * X

    # 4. 构建全空间算符
    Op_A0 = np.kron(A0_single, I)
    Op_A1 = np.kron(A1_single, I)
    Op_B0 = np.kron(I, B0_single)
    Op_B1 = np.kron(I, B1_single)

    Op_A0B0 = np.kron(A0_single, B0_single)
    Op_A0B1 = np.kron(A0_single, B1_single)
    Op_A1B0 = np.kron(A1_single, B0_single)
    Op_A1B1 = np.kron(A1_single, B1_single)

    # 5. 构建量子态 |psi> = cos(theta)|00> + sin(theta)|11>
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    psi = np.array([cos_theta, 0, 0, sin_theta], dtype=complex)

    # 6. 计算期望值函数
    def get_expectation(op, state):
        return np.real(np.vdot(state, op @ state))

    # 计算 8 个量
    exp_A0 = get_expectation(Op_A0, psi)
    exp_A1 = get_expectation(Op_A1, psi)
    exp_B0 = get_expectation(Op_B0, psi)
    exp_B1 = get_expectation(Op_B1, psi)

    exp_A0B0 = get_expectation(Op_A0B0, psi)
    exp_A0B1 = get_expectation(Op_A0B1, psi)
    exp_A1B0 = get_expectation(Op_A1B0, psi)
    exp_A1B1 = get_expectation(Op_A1B1, psi)

    # 按顺序返回: A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1
    return [exp_A0, exp_A1, exp_B0, exp_B1, exp_A0B0, exp_A0B1, exp_A1B0, exp_A1B1]


def analyze_tilted_chsh(alpha):
    """
    原有的打印显示函数，保留用于直接运行此脚本时的展示
    """
    results = get_tilted_expectations(alpha)
    exp_A0, exp_A1, exp_B0, exp_B1, exp_A0B0, exp_A0B1, exp_A1B0, exp_A1B1 = results

    bell_value = alpha * exp_A0 + exp_A0B0 + exp_A0B1 + exp_A1B0 - exp_A1B1
    theo_max = np.sqrt(8 + 2 * alpha ** 2)

    print("-" * 30)
    print(f"1. 当前 alpha 的值: {alpha}")
    print(f"2. 8个可观测量期望值:")
    labels = ["A0", "A1", "B0", "B1", "A0B0", "A0B1", "A1B0", "A1B1"]
    for l, v in zip(labels, results):
        print(f"   <{l}> = {v:.6f}")
    print(f"3. Bell值 (Calc/Theo): {bell_value:.8f} / {theo_max:.8f}")
    print("-" * 30)


if __name__ == "__main__":
    analyze_tilted_chsh(alpha=0.5)