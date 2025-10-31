import numpy as np

# Pauli矩阵定义
I = np.array([[1, 0], [0, 1]])
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])

# 基向量
ket_00 = np.array([[1], [0], [0], [0]])
ket_01 = np.array([[0], [1], [0], [0]])
ket_10 = np.array([[0], [0], [1], [0]])
ket_11 = np.array([[0], [0], [0], [1]])


def calculate_correlations(a0, a1, b0, b1, theta):
    """
    计算量子态的关联函数

    参数:
    a0, a1, b0, b1, theta: 角度值（度）
    """
    # 将角度转换为弧度
    a0_rad = np.deg2rad(a0)
    a1_rad = np.deg2rad(a1)
    b0_rad = np.deg2rad(b0)
    b1_rad = np.deg2rad(b1)
    theta_rad = np.deg2rad(theta)

    # 定义基础量子态
    costheta = np.cos(theta_rad)
    sintheta = np.sin(theta_rad)

    # |theta> = costheta|00> + sintheta|11>
    state_theta = costheta * ket_00 + sintheta * ket_11

    # |theta1> = sintheta|00> + costheta|11>
    state_theta1 = sintheta * ket_00 + costheta * ket_11

    # |theta2> = costheta|01> - sintheta|10>
    state_theta2 = costheta * ket_01 - sintheta * ket_10

    # |theta3> = -sintheta|01> + costheta|10>
    state_theta3 = -sintheta * ket_01 + costheta * ket_10

    # 构建旋转后的量子态 psi_rotate
    cos_a0_2 = np.cos(a0_rad / 2)
    sin_a0_2 = np.sin(a0_rad / 2)
    cos_b0_2 = np.cos(b0_rad / 2)
    sin_b0_2 = np.sin(b0_rad / 2)

    psi_rotate = (cos_a0_2 * cos_b0_2 * state_theta +
                  sin_a0_2 * sin_b0_2 * state_theta1 +
                  cos_a0_2 * sin_b0_2 * state_theta2 +
                  sin_a0_2 * cos_b0_2 * state_theta3)

    # 归一化量子态
    norm = np.sqrt(np.vdot(psi_rotate, psi_rotate))
    psi_rotate = psi_rotate / norm

    # 定义旋转后的测量算子
    # A0_rotate = Z
    A0_rotate = Z

    # A1_rotate = cos(a0+a1)Z + sin(a0+a1)X
    A1_rotate = np.cos(a0_rad + a1_rad) * Z + np.sin(a0_rad + a1_rad) * X

    # B0_rotate = Z
    B0_rotate = Z

    # B1_rotate = cos(b0+b1)Z + sin(b0+b1)X
    B1_rotate = np.cos(b0_rad + b1_rad) * Z + np.sin(b0_rad + b1_rad) * X

    # Tensor product helper
    def tensor_op(op1, op2):
        return np.kron(op1, op2)

    # Expectation value calculation - 修正版本
    def expectation(op):
        # 将列向量转换为行向量进行点积计算
        return (psi_rotate.conj().T @ op @ psi_rotate)[0, 0].real

    # 计算所有关联函数
    corre_A0 = expectation(tensor_op(A0_rotate, I))
    corre_A1 = expectation(tensor_op(A1_rotate, I))
    corre_B0 = expectation(tensor_op(I, B0_rotate))
    corre_B1 = expectation(tensor_op(I, B1_rotate))
    corre_A0B0 = expectation(tensor_op(A0_rotate, B0_rotate))
    corre_A0B1 = expectation(tensor_op(A0_rotate, B1_rotate))
    corre_A1B0 = expectation(tensor_op(A1_rotate, B0_rotate))
    corre_A1B1 = expectation(tensor_op(A1_rotate, B1_rotate))

    return {
        'corre_A0': corre_A0,
        'corre_A1': corre_A1,
        'corre_B0': corre_B0,
        'corre_B1': corre_B1,
        'corre_A0B0': corre_A0B0,
        'corre_A0B1': corre_A0B1,
        'corre_A1B0': corre_A1B0,
        'corre_A1B1': corre_A1B1
    }


# 使用给定的角度值进行计算
a0 = 5  # 度
b0 = 19  # 度
a1 = 25  # 度
b1 = 95  # 度
theta = 25  # 度

results = calculate_correlations(a0, a1, b0, b1, theta)

# 输出结果
print("量子关联函数计算结果:")
print(f"corre_A0 = {results['corre_A0']:.6f}")
print(f"corre_A1 = {results['corre_A1']:.6f}")
print(f"corre_B0 = {results['corre_B0']:.6f}")
print(f"corre_B1 = {results['corre_B1']:.6f}")
print(f"corre_A0B0 = {results['corre_A0B0']:.6f}")
print(f"corre_A0B1 = {results['corre_A0B1']:.6f}")
print(f"corre_A1B0 = {results['corre_A1B0']:.6f}")
print(f"corre_A1B1 = {results['corre_A1B1']:.6f}")