import numpy as np

# Pauli matrices
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

theta = np.pi / 4


def calculate_and_verify(theta):
    """Calculate measurement operators and expectation values using both methods and verify additional expressions"""
    # Precompute trigonometric values
    cos2theta = np.cos(2 * theta)
    sin2theta = np.sin(2 * theta)
    tan2theta = np.tan(2 * theta)

    # Calculate mu
    mu = np.arctan(sin2theta)

    # 旋转后的态 |ψ_r⟩（直接按理论公式写出数组）
    psi_r = np.array([
        0.5 * (np.cos(theta) * (1 + np.cos(mu)) + np.sin(theta) * (1 - np.cos(mu))),
        0.5 * (np.cos(theta) - np.sin(theta)) * np.sin(mu),
        0.5 * (np.cos(theta) - np.sin(theta)) * np.sin(mu),
        0.5 * (np.cos(theta) * (1 - np.cos(mu)) + np.sin(theta) * (1 + np.cos(mu)))
    ])

    # Set measurement angles according to new settings
    a0 = 0
    a1 = 2 * mu
    b0 = mu
    b1 = np.pi / 2 + mu

    # Calculate alpha
    alpha = 2 / np.sqrt(1 + 2 * (tan2theta) ** 2)

    # Method 1: Using matrix operations
    # Calculate measurement operators
    mea_A0 = np.cos(a0) * Z + np.sin(a0) * X
    mea_A1 = np.cos(a1) * Z + np.sin(a1) * X
    mea_B0 = np.cos(b0) * Z + np.sin(b0) * X
    mea_B1 = np.cos(b1) * Z + np.sin(b1) * X


    # method 3:
    mea_A0_r = Z  # A0 旋转后对齐 Z 轴
    mea_A1_r = np.cos(2 * mu) * Z + np.sin(2 * mu) * X
    mea_B0_r = np.cos(mu) * Z + np.sin(mu) * X
    mea_B1_r = - np.sin(mu) * Z + np.cos(mu) * X

    # Tensor product helper
    def tensor_op(op1, op2):
        return np.kron(op1, op2)

    # Expectation value calculation
    def expectation(op):
        return np.dot(psi_r.conj(), np.dot(op, psi_r)).real

    # Calculate correlation values (E terms)
    E00 = expectation(tensor_op(mea_A0, mea_B0))
    E01 = expectation(tensor_op(mea_A0, mea_B1))
    E10 = expectation(tensor_op(mea_A1, mea_B0))
    E11 = expectation(tensor_op(mea_A1, mea_B1))

    # Calculate single-party expectation values
    A0 = expectation(tensor_op(mea_A0, I))  # <A0⊗I>
    A1 = expectation(tensor_op(mea_A1, I))  # <A1⊗I>
    B0 = expectation(tensor_op(I, mea_B0))  # <I⊗B0>
    B1 = expectation(tensor_op(I, mea_B1))  # <I⊗B1>

    # method 3:
    A0_r = expectation(tensor_op(mea_A0_r, I))
    A1_r = expectation(tensor_op(mea_A1_r, I))
    B0_r = expectation(tensor_op(I, mea_B0_r))
    B1_r = expectation(tensor_op(I, mea_B1_r))
    E00_r = expectation(tensor_op(mea_A0_r, mea_B0_r))
    E01_r = expectation(tensor_op(mea_A0_r, mea_B1_r))
    E10_r = expectation(tensor_op(mea_A1_r, mea_B0_r))
    E11_r = expectation(tensor_op(mea_A1_r, mea_B1_r))

    # Prepare results for comparison
    method1 = {
        'A0': A0, 'A1': A1, 'B0': B0, 'B1': B1,
        'E00': E00, 'E01': E01, 'E10': E10, 'E11': E11
    }

    method3 = {
        'A0': A0_r, 'A1': A1_r, 'B0': B0_r, 'B1': B1_r,
        'E00': E00_r, 'E01': E01_r, 'E10': E10_r, 'E11': E11_r
    }

    # Calculate verification expressions for both methods
    expr1_method1 = alpha * A0 + E00 + E01 + E10 - E11

    expr1_method3 = alpha * B0_r + E10_r + E00_r + E11_r - E01_r

    expr2 = np.sqrt(8 + 2 * alpha ** 2)

    return method1, method3, expr1_method1, expr1_method3, expr2, alpha


# Calculate with new measurement settings
method1, method3, expr1_m1, expr1_m3, expr2, alpha = calculate_and_verify(theta)

# Print comparison table
print("Comparison of calculation methods:")
print("{:<10} {:<15} {:<15}".format("Parameter", "Method 1", "Method 3"))
print("-" * 40)
for key in method1:
    val1 = method1[key]
    val3 = method3[key]
    print("{:<10} {:<15.8f} {:<15.8f}".format(key, val1, val3))

print("\nAdditional verification:")
print(f"alpha = {alpha:.8f}")
print(f"alpha*(B0) + E00 + E10 + E11 - E01:")
print(f"  Method 1: {expr1_m1:.8f}")
print(f"  Method 3: {expr1_m3:.8f}")

print(f"\nsqrt(8 + 2*alpha^2) = {expr2:.8f}")