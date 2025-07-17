import numpy as np

# 定义Pauli矩阵
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])


def calculate_correlations(theta, a0, a1, b0=None, b1=None):
    """
    计算量子关联算子

    参数:
    theta: 量子态参数
    a0, a1: Alice测量角度
    b0, b1: Bob测量角度(可选，如果不提供则根据tan(b)=sin(2θ)计算)

    返回:
    correlations: 8个关联算子的字典
    calculated_b: 计算得到的b0,b1(如果未提供)
    """
    # 定义量子态 |psi⟩ = cosθ|00⟩ + sinθ|11⟩
    psi = np.array([np.cos(theta), 0, 0, np.sin(theta)])

    # 如果未提供b0,b1，则根据tan(b)=sin(2θ)计算
    calculated_b = None
    if b0 is None or b1 is None:
        sin2theta = np.sin(2 * theta)
        b0 = np.arctan(sin2theta)
        b1 = np.arctan(sin2theta)
        calculated_b = (b0, b1)

    # 定义测量算符
    A0 = np.cos(a0) * Z + np.sin(a0) * X
    A1 = np.cos(a1) * Z + np.sin(a1) * X
    B0 = np.cos(b0) * Z + np.sin(b0) * X
    B1 = np.cos(b1) * Z - np.sin(b1) * X

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

    return E, calculated_b


# 示例使用
if __name__ == "__main__":
    # 输入参数 (角度以弧度为单位)
    theta = np.pi / 6  # 8,6,3
    a0 = 0
    a1 = np.pi / 2

    # 计算关联算子(不提供b0,b1)
    correlations, (b0, b1) = calculate_correlations(theta, a0, a1)

    # 输出结果
    print("输入参数:")
    print(f"theta = {theta:.4f} rad")
    print(f"a0 = {a0:.4f} rad, a1 = {a1:.4f} rad")
    print("\n计算得到的Bob测量角度:")
    print(f"b0 = {b0:.4f} rad (根据tan(b0) = sin(2θ) = {np.sin(2 * theta):.4f})")
    print(f"b1 = {b1:.4f} rad (根据tan(b1) = sin(2θ) = {np.sin(2 * theta):.4f})")
    print("\n计算结果:")
    for key, value in correlations.items():
        print(f"<psi|{key}|psi> = {value:.6f}")

    # 也可以显式提供b0,b1
    print("\n测试显式提供b0,b1的情况:")
    custom_b0, custom_b1 = 0.3, 0.4
    correlations, _ = calculate_correlations(theta, a0, a1, custom_b0, custom_b1)
    print(f"使用自定义 b0 = {custom_b0:.4f}, b1 = {custom_b1:.4f}")
    for key, value in correlations.items():
        print(f"<psi|{key}|psi> = {value:.6f}")