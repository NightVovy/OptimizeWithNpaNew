import math
import numpy as np
from itertools import product

# Pauli matrices
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)


def calculate_parameters(theta, a0, a1, b0=None, b1=None):
    """Calculate measurement operators and expectation values"""
    # Define quantum state |psi⟩ = cosθ|00⟩ + sinθ|11⟩
    psi = np.array([np.cos(theta), 0, 0, np.sin(theta)])

    # Calculate b0 and b1 if not provided
    sin2theta = np.sin(2 * theta)
    if b0 is None:
        b0 = np.arctan(sin2theta)
    if b1 is None:
        b1 = np.arctan(sin2theta)

    # Calculate measurement operators
    mea_A0 = np.cos(a0) * Z + np.sin(a0) * X
    mea_A1 = np.cos(a1) * Z + np.sin(a1) * X
    mea_B0 = np.cos(b0) * Z + np.sin(b0) * X
    mea_B1 = np.cos(b1) * Z - np.sin(b1) * X

    # Tensor product helper
    def tensor_op(op1, op2):
        return np.kron(op1, op2)

    # Expectation value calculation
    def expectation(op):
        return np.dot(psi.conj(), np.dot(op, psi)).real

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

    return {
        'A0': A0, 'A1': A1, 'B0': B0, 'B1': B1,  # Expectation values
        'E00': E00, 'E01': E01, 'E10': E10, 'E11': E11,
        'mea_A0': mea_A0, 'mea_A1': mea_A1,  # Measurement operators
        'mea_B0': mea_B0, 'mea_B1': mea_B1,
        'b0': b0, 'b1': b1  # Added b0 and b1 to the returned dictionary
    }


def verify_formula(params, s_values=[1, -1], t_values=[1, -1]):
    """Verify the updated formula for all combinations of s and t"""
    A0, A1 = params['A0'], params['A1']
    B0, B1 = params['B0'], params['B1']
    E00, E01 = params['E00'], params['E01']
    E10, E11 = params['E10'], params['E11']

    results = []

    for s, t in product(s_values, t_values):
        # Calculate intermediate terms
        try:
            sA0B0 = (E00 + s * B0) / (1 + s * A0)
            tA1B0 = (E10 + t * B0) / (1 + t * A1)
            sA0B1 = (E01 + s * B1) / (1 + s * A0)
            tA1B1 = (E11 + t * B1) / (1 + t * A1)
        except ZeroDivisionError:
            results.append((s, t, "Undefined (division by zero)"))
            continue

        # Check domain of arcsin
        terms = [sA0B0, tA1B0, sA0B1, -tA1B1]  # Note sign change for last term
        if any(abs(term) > 1 for term in terms):
            results.append((s, t, "Undefined (arcsin domain error)"))
            continue

        # Calculate formula value with updated signs
        value = (math.asin(sA0B0) + math.asin(tA1B0) +
                 math.asin(sA0B1) - math.asin(tA1B1))  # Note sign change

        # Compare with pi
        is_valid = math.isclose(value, math.pi, rel_tol=1e-9)

        results.append((s, t, value, is_valid, {
            '[sA0B0]': sA0B0,
            '[tA1B0]': tA1B0,
            '[sA0B1]': sA0B1,
            '[tA1B1]': tA1B1
        }))

    return results


# Set parameters
theta = np.pi / 6
a0 = 0
a1 = np.pi / 2

# Calculate all needed parameters
params = calculate_parameters(theta, a0, a1)

print("Parameters used:")
print(f"theta = {theta:.6f}")
print(f"a0 = {a0:.6f}")
print(f"a1 = {a1:.6f}")
print(f"b0 = {params['b0']:.6f} (calculated as arctan(sin(2*theta)))")
print(f"b1 = {params['b1']:.6f} (calculated as arctan(sin(2*theta)))")

print("\nExpectation values:")
print(f"A0 = <A0⊗I>: {params['A0']:.6f}")
print(f"A1 = <A1⊗I>: {params['A1']:.6f}")
print(f"B0 = <I⊗B0>: {params['B0']:.6f}")
print(f"B1 = <I⊗B1>: {params['B1']:.6f}")
print(f"E00: {params['E00']:.6f}")
print(f"E01: {params['E01']:.6f}")
print(f"E10: {params['E10']:.6f}")
print(f"E11: {params['E11']:.6f}")

# Calculate and print the additional expectation values
sin2theta = np.sin(2 * theta)
cos2theta = np.cos(2 * theta)

# Use b0 and b1 from the params dictionary
b0 = params['b0']
b1 = params['b1']

A0_5 = cos2theta * np.cos(a0)
A1_5 = cos2theta * np.cos(a1)
B0_5 = cos2theta * np.cos(b0)
B1_5 = cos2theta * np.cos(b1)

E00_5 = np.cos(a0)*np.cos(b0) + sin2theta*np.sin(a0)*np.sin(b0)
E01_5 = np.cos(a0)*np.cos(b1) + sin2theta*np.sin(a0)*np.sin(b1)
E10_5 = np.cos(a1)*np.cos(b0) + sin2theta*np.sin(a1)*np.sin(b0)
E11_5 = np.cos(a1)*np.cos(b1) - sin2theta*np.sin(a1)*np.sin(b1)

print("\n5 paras Expectation values:")
print(f"A0_5: cos(2θ)cos(a0) = {A0_5:.6f}")
print(f"A1_5: cos(2θ)cos(a1) = {A1_5:.6f}")
print(f"B0_5: cos(2θ)cos(b0) = {B0_5:.6f}")
print(f"B1_5: cos(2θ)cos(b1) = {B1_5:.6f}")
print(f"E00_5: cos(a0)cos(b0)+sin(2θ)sin(a0)sin(b0) = {E00_5:.6f}")
print(f"E01_5: cos(a0)cos(b1)+sin(2θ)sin(a0)sin(b1) = {E01_5:.6f}")
print(f"E10_5: cos(a1)cos(b0)+sin(2θ)sin(a1)sin(b0) = {E10_5:.6f}")
print(f"E11_5: cos(a1)cos(b1)-sin(2θ)sin(a1)sin(b1) = {E11_5:.6f}")

print("\nVerification results for updated formula:")
print("arcsin[sA0B0] + arcsin[tA1B0] + arcsin[sA0B1] - arcsin[tA1B1] = pi")
results = verify_formula(params)
for r in results:
    s, t, *rest = r
    print(f"\nCombination s={s}, t={t}:")
    if isinstance(rest[0], str):
        print(rest[0])
    else:
        value, is_valid, terms = rest
        print(f"Formula value: {value:.6f}")
        print(f"Equals pi: {is_valid}")
        print("Intermediate terms:")
        for term, val in terms.items():
            print(f"{term}: {val:.6f}")