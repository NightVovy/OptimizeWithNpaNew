import numpy as np
from numpy.ma.core import arctan

# 定义Pauli矩阵
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])


def calculate_correlations(theta_deg, a0_deg, a1_deg, b0_deg, b1_deg):
    # 将角度转换为弧度
    theta = np.deg2rad(theta_deg)
    a0 = np.deg2rad(a0_deg)
    a1 = np.deg2rad(a1_deg)
    b0 = np.deg2rad(b0_deg)
    b1 = np.deg2rad(b1_deg)

    # 定义量子态 |psi⟩ = cosθ|00⟩ + sinθ|11⟩
    psi = np.array([np.cos(theta), 0, 0, np.sin(theta)])

    # 定义测量算符
    A0 = np.cos(a0) * Z + np.sin(a0) * X
    A1 = np.cos(a1) * Z + np.sin(a1) * X
    B0 = np.cos(b0) * Z + np.sin(b0) * X
    B1 = np.cos(b1) * Z + np.sin(b1) * X

    # 计算张量积算符
    def tensor_op(op1, op2):
        return np.kron(op1, op2)

    # 计算期望值
    def expectation_value(op):
        return np.dot(psi.conj(), np.dot(op, psi)).real

    # 计算8个关联算子
    E = {}
    E['A0B0'] = expectation_value(tensor_op(A0, B0))
    E['A0B1'] = expectation_value(tensor_op(A0, B1))
    E['A1B0'] = expectation_value(tensor_op(A1, B0))
    E['A1B1'] = expectation_value(tensor_op(A1, B1))
    E['A0I'] = expectation_value(tensor_op(A0, I))
    E['A1I'] = expectation_value(tensor_op(A1, I))
    E['IB0'] = expectation_value(tensor_op(I, B0))
    E['IB1'] = expectation_value(tensor_op(I, B1))

    return E


# 示例使用
if __name__ == "__main__":
    # 输入参数 (以角度为单位)
    theta_deg = 25  # 45度
    a0_deg = 5
    a1_deg = 95
    b0_deg = 25
    b1_deg = 19

    # 计算关联算子
    correlations = calculate_correlations(theta_deg, a0_deg, a1_deg, b0_deg, b1_deg)

    # 输出结果
    print("输入参数:")
    print(f"theta = {np.deg2rad(theta_deg):.4f} rad ({theta_deg}°)")
    print(f"a0 = {np.deg2rad(a0_deg):.4f} rad ({a0_deg}°), a1 = {np.deg2rad(a1_deg):.4f} rad ({a1_deg}°)")
    print(f"b0 = {np.deg2rad(b0_deg):.4f} rad ({b0_deg}°), b1 = {np.deg2rad(b1_deg):.4f} rad ({b1_deg}°)")
    print("\n计算结果:")
    for key, value in correlations.items():
        print(f"<psi|{key}|psi> = {value:.6f}")