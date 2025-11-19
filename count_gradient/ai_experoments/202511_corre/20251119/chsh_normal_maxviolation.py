# chsh_normal_maxviolation.py

import numpy as np
import sys

# --- 1. 定义 Pauli 矩阵和单位矩阵 ---
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)
sqrt2 = np.sqrt(2)

# --- 2. 定义测量算符 ---
# 测量设置：A0=Z, A1=X, B0=(Z+X)/sqrt2, B1=(Z-X)/sqrt2
mea_A0 = Z
mea_A1 = X
mea_B0 = (Z + X) / sqrt2
mea_B1 = (Z - X) / sqrt2

# --- 3. 定义目标量子态 $|\psi\rangle$ ---
psi_r = (1 / sqrt2) * np.array([1, 0, 0, 1])


# --- 4. 辅助函数 ---
def tensor_op(op1, op2):
    """计算两个算符的张量积 np.kron(op1, op2)"""
    return np.kron(op1, op2)


def expectation(op):
    """计算算符 op 对量子态 psi_r 的期望值 $\langle\psi|op|\psi\rangle$"""
    return np.dot(psi_r.conj(), np.dot(op, psi_r)).real


# --- 5. 封装计算逻辑并返回结果 ---
def get_chsh_correlations():
    # 计算 CHSH 所需的 8 个测量结果 (期望值)

    # 单方期望值 (Ai 和 Bi 项)
    corre_A0 = expectation(tensor_op(mea_A0, I))  # $\langle A_0 \otimes I \rangle$
    corre_A1 = expectation(tensor_op(mea_A1, I))  # $\langle A_1 \otimes I \rangle$
    corre_B0 = expectation(tensor_op(I, mea_B0))  # $\langle I \otimes B_0 \rangle$
    corre_B1 = expectation(tensor_op(I, mea_B1))  # $\langle I \otimes B_1 \rangle$

    # 关联值 (AiBj 项)
    corre_A0B0 = expectation(tensor_op(mea_A0, mea_B0))  # $\langle A_0 \otimes B_0 \rangle$
    corre_A0B1 = expectation(tensor_op(mea_A0, mea_B1))  # $\langle A_0 \otimes B_1 \rangle$
    corre_A1B0 = expectation(tensor_op(mea_A1, mea_B0))  # $\langle A_1 \otimes B_0 \rangle$
    corre_A1B1 = expectation(tensor_op(mea_A1, mea_B1))  # $\langle A_1 \otimes B_1 \rangle$

    # 按照 [A0, A1, B0, B1, A0B0, A0B1, A1B0, A1B1] 的顺序返回 NumPy 数组
    return np.array([
        corre_A0, corre_A1, corre_B0, corre_B1,
        corre_A0B0, corre_A0B1, corre_A1B0, corre_A1B1
    ])


if __name__ == "__main__":
    # 如果直接运行此文件，则执行原始的打印和计算逻辑
    correlations = get_chsh_correlations()

    # 将数组展开成字典方便打印
    keys = ['corre_A0', 'corre_A1', 'corre_B0', 'corre_B1',
            'corre_A0B0', 'corre_A0B1', 'corre_A1B0', 'corre_A1B1']
    method1 = dict(zip(keys, correlations))

    print("### 8 个测量期望值 ###")
    for key, value in method1.items():
        print(f"<{key}>: {np.round(value, 8)}")

    chsh_value = method1['corre_A0B0'] + method1['corre_A0B1'] + method1['corre_A1B0'] - method1['corre_A1B1']
    print("\n--- CHSH = 计算A0B0 + A0B1 + A1B0 - A1B1 ---")
    print(f"CHSH 值为: {np.round(chsh_value, 8)}")