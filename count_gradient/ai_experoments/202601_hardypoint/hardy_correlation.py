import numpy as np


def calculate_hardy_correlations():
    """
    计算 Hardy Point 下的量子关联。

    Returns:
        list: 包含8个correlation值，顺序为:
              [A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1]
    """
    # 1. 定义基础常数和 Pauli 矩阵
    # ---------------------------------------------------------
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    # 题目给定的参数 a
    a = np.sqrt(np.sqrt(5) - 2)

    # 2. 构建量子态 |psi_H>
    # ---------------------------------------------------------
    # Basis vectors
    ket_0 = np.array([1, 0], dtype=complex)
    ket_1 = np.array([0, 1], dtype=complex)

    # Tensor product basis states
    ket_01 = np.kron(ket_0, ket_1)
    ket_10 = np.kron(ket_1, ket_0)
    ket_11 = np.kron(ket_1, ket_1)

    # Coefficients
    coeff_entangled = np.sqrt((1 - a ** 2) / 2)

    # |psi_H>
    psi_H = coeff_entangled * (ket_01 + ket_10) + a * ket_11

    # 归一化 (以防万一)
    psi_H = psi_H / np.linalg.norm(psi_H)

    # 3. 构建单体测量算符
    # ---------------------------------------------------------
    # A0 = B0 = 2a*sx + sqrt(1-4a^2)*sz
    term_z = np.sqrt(1 - 4 * (a ** 2))
    Op_single_0 = 2 * a * sigma_x + term_z * sigma_z

    # A1 = B1 = -sz
    Op_single_1 = -sigma_z

    # Alice's operators
    A0_single = Op_single_0
    A1_single = Op_single_1

    # Bob's operators
    B0_single = Op_single_0
    B1_single = Op_single_1

    # 4. 计算期望值
    # ---------------------------------------------------------
    def expect(op_matrix, state_vector):
        val = state_vector.conj().T @ op_matrix @ state_vector
        return val.real

    # 按指定顺序计算 8 项
    # Order: A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1

    # Marginals (单体)
    val_A0 = expect(np.kron(A0_single, identity), psi_H)
    val_A1 = expect(np.kron(A1_single, identity), psi_H)
    val_B0 = expect(np.kron(identity, B0_single), psi_H)
    val_B1 = expect(np.kron(identity, B1_single), psi_H)

    # Correlations (关联)
    val_A0B0 = expect(np.kron(A0_single, B0_single), psi_H)
    val_A0B1 = expect(np.kron(A0_single, B1_single), psi_H)
    val_A1B0 = expect(np.kron(A1_single, B0_single), psi_H)
    val_A1B1 = expect(np.kron(A1_single, B1_single), psi_H)

    # 返回列表
    return [val_A0, val_A1, val_B0, val_B1, val_A0B0, val_A0B1, val_A1B0, val_A1B1]


if __name__ == "__main__":
    # 本地测试代码
    correlations = calculate_hardy_correlations()

    labels = ["A0", "A1", "B0", "B1", "A0B0", "A0B1", "A1B0", "A1B1"]

    print("-" * 30)
    print(" 计算结果 (For External Call) ")
    print("-" * 30)
    for label, value in zip(labels, correlations):
        print(f"{label:<5}: {value:.8f}")
    print("-" * 30)