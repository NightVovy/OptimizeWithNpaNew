import math
import numpy as np
from itertools import product
import random

# Pauli matrices
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)


def calculate_parameters(theta, a0, a1, b0=None, b1=None):
    """Calculate measurement operators and expectation values"""
    psi = np.array([np.cos(theta), 0, 0, np.sin(theta)])
    sin2theta = np.sin(2 * theta)

    if b0 is None:
        b0 = np.arctan(sin2theta)
    if b1 is None:
        b1 = (3 * np.pi / 2) + np.arctan(sin2theta)

    mea_A0 = np.cos(a0) * Z + np.sin(a0) * X
    mea_A1 = np.cos(a1) * Z + np.sin(a1) * X
    mea_B0 = np.cos(b0) * Z + np.sin(b0) * X
    mea_B1 = np.cos(b1) * Z + np.sin(b1) * X

    def tensor_op(op1, op2):
        return np.kron(op1, op2)

    def expectation(op):
        return np.dot(psi.conj(), np.dot(op, psi)).real

    E00 = expectation(tensor_op(mea_A0, mea_B0))
    E01 = expectation(tensor_op(mea_A0, mea_B1))
    E10 = expectation(tensor_op(mea_A1, mea_B0))
    E11 = expectation(tensor_op(mea_A1, mea_B1))

    A0 = expectation(tensor_op(mea_A0, I))
    A1 = expectation(tensor_op(mea_A1, I))
    B0 = expectation(tensor_op(I, mea_B0))
    B1 = expectation(tensor_op(I, mea_B1))

    return {
        'A0': A0, 'A1': A1, 'B0': B0, 'B1': B1,
        'E00': E00, 'E01': E01, 'E10': E10, 'E11': E11,
        'mea_A0': mea_A0, 'mea_A1': mea_A1,
        'mea_B0': mea_B0, 'mea_B1': mea_B1,
        'b0': b0, 'b1': b1
    }


def verify_formula(params, s_values=[1, -1], t_values=[1, -1]):
    """Verify the updated formula with full intermediate results"""
    A0, A1 = params['A0'], params['A1']
    B0, B1 = params['B0'], params['B1']
    E00, E01 = params['E00'], params['E01']
    E10, E11 = params['E10'], params['E11']

    results = []
    for s, t in product(s_values, t_values):
        try:
            sA0B0 = (E00 + s * B0) / (1 + s * A0)
            tA1B0 = (E10 + t * B0) / (1 + t * A1)
            sA0B1 = (E01 + s * B1) / (1 + s * A0)
            tA1B1 = (E11 + t * B1) / (1 + t * A1)
        except ZeroDivisionError:
            results.append((s, t, "Undefined (division by zero)", None, None, None, None))
            continue

        terms = [sA0B0, tA1B0, sA0B1, -tA1B1]
        if any(abs(term) > 1 for term in terms):
            results.append((s, t, "Undefined (arcsin domain error)", sA0B0, tA1B0, sA0B1, tA1B1))
            continue

        value = (math.asin(sA0B0) + math.asin(tA1B0) +
                 math.asin(sA0B1) - math.asin(tA1B1))
        is_valid = math.isclose(value, math.pi, rel_tol=1e-9)

        results.append((s, t, value, is_valid, sA0B0, tA1B0, sA0B1, tA1B1))
    return results


def randomize_a0(a0, a1, iteration):
    """Randomly change a0 with guaranteed positive results"""
    max_abs = abs(a1)

    # Determine change range
    if iteration == 0:
        min_change, max_change = 0.05, 0.2
    else:
        min_change, max_change = 0.06, 0.2

    # Calculate maximum possible increase (since we only allow positive a0)
    remaining_range = max_abs - a0  # a0 is always positive now
    actual_max_change = min(max_change, remaining_range)

    # Ensure we can meet the minimum requirement
    if actual_max_change < min_change:
        new_a0 = max_abs  # Go directly to maximum
    else:
        # Generate positive change within required range
        change = random.uniform(min_change, actual_max_change)
        new_a0 = a0 + change

    # Final boundary check
    if new_a0 > max_abs:
        new_a0 = max_abs

    return new_a0


def run_simulation(theta, a0, a1, iterations=5):
    """Run the simulation with a0 strictly positive"""
    # Ensure initial a0 is positive
    a0 = abs(a0)

    for i in range(iterations):
        print(f"\n=== Iteration {i + 1} ===")
        print(f"Starting a0: {a0:.6f} (0 < a0 ≤ {abs(a1):.6f})")

        params = calculate_parameters(theta, a0, a1)

        print("\nExpectation values:")
        print(f"A0: {params['A0']:.6f}  A1: {params['A1']:.6f}")
        print(f"B0: {params['B0']:.6f}  B1: {params['B1']:.6f}")
        print(f"E00: {params['E00']:.6f}  E01: {params['E01']:.6f}")
        print(f"E10: {params['E10']:.6f}  E11: {params['E11']:.6f}")

        print("\nFormula verification:")
        results = verify_formula(params)
        for r in results:
            s, t, *rest = r
            print(f"\nFor s={s}, t={t}:")

            if isinstance(rest[0], str):
                print(rest[0])
            else:
                value, is_valid, sA0B0, tA1B0, sA0B1, tA1B1 = rest
                print(f"Final value: {value:.6f} {'≈ π' if is_valid else '≠ π'}")
                print("Intermediate terms:")
                print(f"[sA0B0]: {sA0B0:.6f}")
                print(f"[tA1B0]: {tA1B0:.6f}")
                print(f"[sA0B1]: {sA0B1:.6f}")
                print(f"[tA1B1]: {tA1B1:.6f}")

        old_a0 = a0
        a0 = randomize_a0(a0, a1, i)
        change = a0 - old_a0  # Always positive
        print(f"\nApplied change: +{change:.6f}")
        print(f"New a0: {a0:.6f} (0 < a0 ≤ {abs(a1):.6f})")


# Set initial parameters
theta = np.pi / 6
initial_a0 = 0  # Will be converted to positive
a1 = np.pi / 2

# Run the simulation
run_simulation(theta, initial_a0, a1, iterations=5)