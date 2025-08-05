import numpy as np

# Pauli matrices
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

theta = np.pi / 6


def calculate_and_verify(theta):
    """Calculate measurement operators and expectation values using both methods and verify additional expressions"""
    # Define quantum state |psi⟩ = cosθ|00⟩ + sinθ|11⟩
    psi = np.array([np.cos(theta), 0, 0, np.sin(theta)])


    # Precompute trigonometric values
    cos2theta = np.cos(2 * theta)
    sin2theta = np.sin(2 * theta)
    tan2theta = np.tan(2 * theta)

    # Calculate mu
    mu = np.arctan(sin2theta)

    # Set measurement angles according to new settings
    a0 = 0
    a1 = 2 * mu
    b0 = mu
    b1 = np.pi / 2 + mu

    # Calculate alpha
    alpha = 2 / np.sqrt(1 + 2 * (tan2theta) ** 2)

    psi_prime = np.array([
        np.cos(theta) * np.cos(mu / 2) ** 2 + np.sin(theta) * np.sin(mu / 2) ** 2,
        (np.sin(theta) - np.cos(theta)) * np.sin(mu / 2) * np.cos(mu / 2),
        (np.sin(theta) - np.cos(theta)) * np.sin(mu / 2) * np.cos(mu / 2),
        np.cos(theta) * np.sin(mu / 2) ** 2 + np.sin(theta) * np.cos(mu / 2) ** 2
    ])

    # Method 1: Using matrix operations
    # Calculate measurement operators
    mea_A0 = np.cos(a0) * Z + np.sin(a0) * X
    mea_A1 = np.cos(a1) * Z + np.sin(a1) * X
    mea_B0 = np.cos(b0) * Z + np.sin(b0) * X
    mea_B1 = np.cos(b1) * Z + np.sin(b1) * X

    # Tensor product helper
    def tensor_op(op1, op2):
        return np.kron(op1, op2)

    # Expectation value calculation
    def expectation(op):
        return np.dot(psi_prime.conj(), np.dot(op, psi_prime)).real

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

    # Method 2: Using analytical formulas
    A0_2 = cos2theta * np.cos(a0)
    A1_2 = cos2theta * np.cos(a1)
    B0_2 = cos2theta * np.cos(b0)
    B1_2 = cos2theta * np.cos(b1)

    E00_2 = np.cos(a0) * np.cos(b0) + sin2theta * np.sin(a0) * np.sin(b0)
    E01_2 = np.cos(a0) * np.cos(b1) + sin2theta * np.sin(a0) * np.sin(b1)
    E10_2 = np.cos(a1) * np.cos(b0) + sin2theta * np.sin(a1) * np.sin(b0)
    E11_2 = np.cos(a1) * np.cos(b1) + sin2theta * np.sin(a1) * np.sin(b1)

    # Prepare results for comparison
    method1 = {
        'A0': A0, 'A1': A1, 'B0': B0, 'B1': B1,
        'E00': E00, 'E01': E01, 'E10': E10, 'E11': E11
    }

    method2 = {
        'A0': A0_2, 'A1': A1_2, 'B0': B0_2, 'B1': B1_2,
        'E00': E00_2, 'E01': E01_2, 'E10': E10_2, 'E11': E11_2
    }

    # Calculate verification expressions for both methods
    expr1_method1 = alpha * A0 + E00 + E01 + E10 - E11
    expr1_method2 = alpha * A0_2 + E00_2 + E01_2 + E10_2 - E11_2

    expr2 = np.sqrt(8 + 2 * alpha ** 2)

    return method1, method2, expr1_method1, expr1_method2, expr2, alpha


# Calculate with new measurement settings
method1, method2, expr1_m1, expr1_m2, expr2, alpha = calculate_and_verify(theta)

# Print comparison table
print("Comparison of two calculation methods:")
print("{:<10} {:<15} {:<15} {:<10}".format("Parameter", "Method 1", "Method 2", "Difference"))
print("-" * 55)
for key in method1:
    val1 = method1[key]
    val2 = method2[key]
    diff = abs(val1 - val2)
    print("{:<10} {:<15.8f} {:<15.8f} {:<10.2e}".format(key, val1, val2, diff))

print("\nAdditional verification:")
print(f"alpha = {alpha:.8f}")
print(f"alpha*(?) + E00 + E01 + E10 - E11:")
print(f"  Method 1: {expr1_m1:.8f}")
print(f"  Method 2: {expr1_m2:.8f}")
print(f"  Difference: {abs(expr1_m1 - expr1_m2):.2e}")
print(f"\nsqrt(8 + 2*alpha^2) = {expr2:.8f}")