import numpy as np
import math

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
    alpha: 计算得到的alpha值
    sqrt_value: 计算得到的sqrt(8 + 2*alpha²)值
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

    # 计算alpha值
    alpha = 2 / math.sqrt(1 + 2 * (math.tan(2 * theta)) ** 2)

    # 计算sqrt(8 + 2*alpha²)值
    sqrt_value = math.sqrt(8 + 2 * (alpha ** 2))

    # 定义测量算符
    A0 = np.cos(a0) * Z + np.sin(a0) * X
    A1 = np.cos(a1) * Z + np.sin(a1) * X
    B0 = np.cos(b0) * Z + np.sin(b0) * X
    B1 = - np.cos(b1) * Z + np.sin(b1) * X

    # 计算张量积算符
    def tensor_op(op1, op2):
        return np.kron(op1, op2)

    # 计算期望值
    def expectation_value(op):
        return np.dot(psi.conj(), np.dot(op, psi)).real

    # 计算8个关联算子
    correlations = {}
    correlations['EA0B0'] = expectation_value(tensor_op(A0, B0))
    correlations['EA0B1'] = expectation_value(tensor_op(A0, B1))
    correlations['EA1B0'] = expectation_value(tensor_op(A1, B0))
    correlations['EA1B1'] = expectation_value(tensor_op(A1, B1))
    correlations['EA0'] = expectation_value(tensor_op(A0, I))
    correlations['EA1'] = expectation_value(tensor_op(A1, I))
    correlations['EB0'] = expectation_value(tensor_op(I, B0))
    correlations['EB1'] = expectation_value(tensor_op(I, B1))

    return (
        expectation_value(tensor_op(A0, I)),  # EA0
        expectation_value(tensor_op(A1, I)),  # EA1
        expectation_value(tensor_op(I, B0)),  # EB0
        expectation_value(tensor_op(I, B1)),  # EB1
        expectation_value(tensor_op(A0, B0)), # EA0B0
        expectation_value(tensor_op(A0, B1)), # EA0B1
        expectation_value(tensor_op(A1, B0)), # EA1B0
        expectation_value(tensor_op(A1, B1)), # EA1B1
        # 'correlations': correlations,
        (b0, b1),
        alpha,
        sqrt_value
    )


# 示例使用

# 输入参数 (角度以弧度为单位)
theta = np.pi / 5 # 6
a0 = 0
a1 = np.pi / 2

    # 计算关联算子(不提供b0,b1)
EA0, EA1, EB0, EB1, EA0B0, EA0B1, EA1B0, EA1B1, (b0, b1), alpha, sqrt_value = calculate_correlations(theta, a0, a1)

    # 计算alpha*A0 + A0B0 - A0B1 + A1B0 + A1B1
alpha_A0_plus_correlations = alpha * EA0 + EA0B0 - EA0B1 + \
                                 EA1B0 + EA1B1

    # 输出结果
print("输入参数:")
print(f"theta = {theta:.4f} rad")
print(f"a0 = {a0:.4f} rad, a1 = {a1:.4f} rad")
print("\n计算得到的Bob测量角度:")
print(f"b0 = {b0:.4f} rad (根据tan(b0) = sin(2θ) = {np.sin(2 * theta):.4f})")
print(f"b1 = {b1:.4f} rad (根据tan(b1) = sin(2θ) = {np.sin(2 * theta):.4f})")
print("\n计算得到的alpha值:")
print(f"alpha = {alpha:.6f}")
print("\n计算alpha*A0 + A0B0 - A0B1 + A1B0 + A1B1的值:")
print(f"alpha*A0 + A0B0 - A0B1 + A1B0 + A1B1 = {alpha_A0_plus_correlations:.6f}")
print("\n计算得到的sqrt(8 + 2*alpha²)值:")
print(f"sqrt(8 + 2*alpha²) = {sqrt_value:.6f}")
print("单算符期望值:")
print(f"<A0> = {EA0:.4f}")
print(f"<A1> = {EA1:.4f}")
print(f"<B0> = {EB0:.4f}")
print(f"<B1> = {EB1:.4f}")

print("\n联合期望值:")
print(f"<A0B0> = {EA0B0:.4f}")
print(f"<A0B1> = {EA0B1:.4f}")
print(f"<A1B0> = {EA1B0:.4f}")
print(f"<A1B1> = {EA1B1:.4f}")