import numpy as np

# Pauli 矩阵定义
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])

# 定义角度（单位：弧度）
a0 = 0
a1 = np.pi / 2
b0 = np.pi / 4
b1 = 3 * np.pi / 4
theta = np.pi / 4  # psi 的参数

# 计算 cos 和 sin
cosa0, sina0 = np.cos(a0), np.sin(a0)
cosa1, sina1 = np.cos(a1), np.sin(a1)
cosb0, sinb0 = np.cos(b0), np.sin(b0)
cosb1, sinb1 = np.cos(b1), np.sin(b1)
costheta, sintheta = np.cos(theta), np.sin(theta)

# 构造测量算符
A0 = cosa0 * Z + sina0 * X
A1 = cosa1 * Z + sina1 * X
B0 = cosb0 * Z + sinb0 * X
B1 = cosb1 * Z + sinb1 * X

# 构造态 |psi> = cos(theta) |00> + sin(theta) |11>
zero = np.array([[1], [0]])
one = np.array([[0], [1]])
psi = costheta * np.kron(zero, zero) + sintheta * np.kron(one, one)
psi_dag = psi.conj().T

# 计算期望值 <psi| A⊗B |psi>
def expectation(A, B):
    return float(np.real(psi_dag @ np.kron(A, B) @ psi))

E00 = expectation(A0, B0)
E01 = expectation(A0, B1)
E10 = expectation(A1, B0)
E11 = expectation(A1, B1)

# 打印结果
print(f"cosa0 = {cosa0:.6f}, sina0 = {sina0:.6f}")
print(f"A0 = {cosa0:.6f} Z + {sina0:.6f} X\n")

print(f"cosa1 = {cosa1:.6f}, sina1 = {sina1:.6f}")
print(f"A1 = {cosa1:.6f} Z + {sina1:.6f} X\n")

print(f"cosb0 = {cosb0:.6f}, sinb0 = {sinb0:.6f}")
print(f"B0 = {cosb0:.6f} Z + {sinb0:.6f} X\n")

print(f"cosb1 = {cosb1:.6f}, sinb1 = {sinb1:.6f}")
print(f"B1 = {cosb1:.6f} Z + {sinb1:.6f} X\n")

print("期望值结果：")
print(f"<ψ|A0⊗B0|ψ> = {E00:.6f}")
print(f"<ψ|A0⊗B1|ψ> = {E01:.6f}")
print(f"<ψ|A1⊗B0|ψ> = {E10:.6f}")
print(f"<ψ|A1⊗B1|ψ> = {E11:.6f}")
