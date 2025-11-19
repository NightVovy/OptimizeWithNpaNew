import numpy as np

# --- 1. 定义 Pauli 矩阵和单位矩阵 ---
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)
theta = np.pi / 8

# --- 2. 定义测量算符 ---
# 测量设置：A0=B0=Z, A1=B1=X
mea_A0 = Z
mea_B0 = Z
mea_A1 = X
mea_B1 = X

# --- 3. 定义目标量子态 $|\psi\rangle$ ---

# 基础 Bell 态
psi_phi_minus = (1 / np.sqrt(2)) * np.array([1, 0, 0, -1])  # |\Phi^-\rangle
psi_psi_plus = (1 / np.sqrt(2)) * np.array([0, 1, 1, 0])    # |\Psi^+\rangle

# 目标态 $|\psi\rangle = cos(pi/8) |\Phi^-\rangle + sin(pi/8) |\Psi^+\rangle$
psi_r = np.cos(theta) * psi_phi_minus + np.sin(theta) * psi_psi_plus

# --- 4. 辅助函数 ---

# Tensor product helper (张量积)
def tensor_op(op1, op2):
    """计算两个算符的张量积 np.kron(op1, op2)"""
    return np.kron(op1, op2)

# Expectation value calculation (期望值计算)
def expectation(op):
    """计算算符 op 对量子态 psi_r 的期望值 $\langle\psi|op|\psi\rangle$"""
    return np.dot(psi_r.conj(), np.dot(op, psi_r)).real

# --- 5. 计算 CHSH 所需的 8 个测量结果 (期望值) ---

# (1) 计算关联值 (AiBj 项) - 名称已按要求修改
corre_A0B0 = expectation(tensor_op(mea_A0, mea_B0)) # $\langle A_0 \otimes B_0 \rangle$
corre_A0B1 = expectation(tensor_op(mea_A0, mea_B1)) # $\langle A_0 \otimes B_1 \rangle$
corre_A1B0 = expectation(tensor_op(mea_A1, mea_B0)) # $\langle A_1 \otimes B_0 \rangle$
corre_A1B1 = expectation(tensor_op(mea_A1, mea_B1)) # $\langle A_1 \otimes B_1 \rangle$

# (2) 计算单方期望值 (Ai 和 Bi 项) - **已添加 corre_ 前缀**
corre_A0 = expectation(tensor_op(mea_A0, I))  # $\langle A_0 \otimes I \rangle$
corre_A1 = expectation(tensor_op(mea_A1, I))  # $\langle A_1 \otimes I \rangle$
corre_B0 = expectation(tensor_op(I, mea_B0))  # $\langle I \otimes B_0 \rangle$
corre_B1 = expectation(tensor_op(I, mea_B1))  # $\langle I \otimes B_1 \rangle$
# --- 6. 整理输出结果 ---

# 整理成字典，键名已按要求修改
method1 = {
    'corre_A0': corre_A0, 'corre_A1': corre_A1,
    'corre_B0': corre_B0, 'corre_B1': corre_B1,
    'corre_A0B0': corre_A0B0, 'corre_A0B1': corre_A0B1,
    'corre_A1B0': corre_A1B0, 'corre_A1B1': corre_A1B1
}

# 打印所有 8 个期望值
print("### 8 个测量期望值 ###")
for key, value in method1.items():
    print(f"<{key}>: {np.round(value, 8)}")

# --- 7. 计算 CHSH 值 S ---

# CHSH 不等式 $\mathcal{S} = \langle A_0 B_0 \rangle + \langle A_0 B_1 \rangle + \langle A_1 B_0 \rangle - \langle A_1 B_1 \rangle$
chsh_value = corre_A0B0 + corre_A0B1 + corre_A1B0 - corre_A1B1

print("\n--- CHSH = 计算A0B0 + A0B1 + A1B0 - A1B1 ---")
print(f"CHSH 值为: {np.round(chsh_value, 8)}")