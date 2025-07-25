import math
import numpy as np
from scipy.linalg import svd, qr
from sympy import symbols, Matrix, asin, pi, diff, N, simplify
from itertools import product

# Pauli matrices
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)


def calculate_parameters(theta, a0, a1, b0=None, b1=None):
    """Calculate measurement operators and expectation values"""
    psi = np.array([np.cos(theta), 0, 0, np.sin(theta)])

    if b0 is None or b1 is None:
        sin2theta = np.sin(2 * theta)
        b0 = np.arctan(sin2theta)
        b1 = np.arctan(sin2theta)

    mea_A0 = np.cos(a0) * Z + np.sin(a0) * X
    mea_A1 = np.cos(a1) * Z + np.sin(a1) * X
    mea_B0 = np.cos(b0) * Z + np.sin(b0) * X
    mea_B1 = np.cos(b1) * Z - np.sin(b1) * X

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
        'E00': E00, 'E01': E01, 'E10': E10, 'E11': E11
    }


def compute_all_variables(params):
    """Compute all 8 variables (plus and minus versions)"""
    A0, A1, B0, B1 = params['A0'], params['A1'], params['B0'], params['B1']
    E00, E01, E10, E11 = params['E00'], params['E01'], params['E10'], params['E11']

    sA0B0plus = (E00 + B0) / (1 + A0)
    sA0B0minus = (E00 - B0) / (1 - A0)

    tA1B0plus = (E10 + B0) / (1 + A1)
    tA1B0minus = (E10 - B0) / (1 - A1)

    sA0B1plus = (E01 + B1) / (1 + A0)
    sA0B1minus = (E01 - B1) / (1 - A0)

    tA1B1plus = (E11 + B1) / (1 + A1)
    tA1B1minus = (E11 - B1) / (1 - A1)

    return {
        'sA0B0plus': sA0B0plus, 'sA0B0minus': sA0B0minus,
        'tA1B0plus': tA1B0plus, 'tA1B0minus': tA1B0minus,
        'sA0B1plus': sA0B1plus, 'sA0B1minus': sA0B1minus,
        'tA1B1plus': tA1B1plus, 'tA1B1minus': tA1B1minus
    }


def verify_equations(vars_dict):
    """Verify all 4 equations with numerical values"""
    results = []

    # Equation 1: s=1,t=1
    try:
        val1 = (math.asin(vars_dict['sA0B0plus']) +
                math.asin(vars_dict['tA1B0plus']) +
                math.asin(vars_dict['sA0B1plus']) -
                math.asin(vars_dict['tA1B1plus']))
        eq1_valid = math.isclose(val1, math.pi, rel_tol=1e-9)
    except (ValueError, ZeroDivisionError):
        val1 = "Undefined"
        eq1_valid = False

    # Equation 2: s=1,t=-1
    try:
        val2 = (math.asin(vars_dict['sA0B0plus']) +
                math.asin(vars_dict['tA1B0minus']) +
                math.asin(vars_dict['sA0B1plus']) -
                math.asin(vars_dict['tA1B1minus']))
        eq2_valid = math.isclose(val2, math.pi, rel_tol=1e-9)
    except (ValueError, ZeroDivisionError):
        val2 = "Undefined"
        eq2_valid = False

    # Equation 3: s=-1,t=1
    try:
        val3 = (math.asin(vars_dict['sA0B0minus']) +
                math.asin(vars_dict['tA1B0plus']) +
                math.asin(vars_dict['sA0B1minus']) -
                math.asin(vars_dict['tA1B1plus']))
        eq3_valid = math.isclose(val3, math.pi, rel_tol=1e-9)
    except (ValueError, ZeroDivisionError):
        val3 = "Undefined"
        eq3_valid = False

    # Equation 4: s=-1,t=-1
    try:
        val4 = (math.asin(vars_dict['sA0B0minus']) +
                math.asin(vars_dict['tA1B0minus']) +
                math.asin(vars_dict['sA0B1minus']) -
                math.asin(vars_dict['tA1B1minus']))
        eq4_valid = math.isclose(val4, math.pi, rel_tol=1e-9)
    except (ValueError, ZeroDivisionError):
        val4 = "Undefined"
        eq4_valid = False

    return [
        (1, 1, val1, eq1_valid),
        (1, -1, val2, eq2_valid),
        (-1, 1, val3, eq3_valid),
        (-1, -1, val4, eq4_valid)
    ]


def check_linear_independence():
    """Symbolic method to check linear independence"""
    sA0B0_p, sA0B0_m = symbols('sA0B0_p sA0B0_m')
    tA1B0_p, tA1B0_m = symbols('tA1B0_p tA1B0_m')
    sA0B1_p, sA0B1_m = symbols('sA0B1_p sA0B1_m')
    tA1B1_p, tA1B1_m = symbols('tA1B1_p tA1B1_m')

    eq1 = asin(sA0B0_p) + asin(tA1B0_p) + asin(sA0B1_p) - asin(tA1B1_p) - pi
    eq2 = asin(sA0B0_p) + asin(tA1B0_m) + asin(sA0B1_p) - asin(tA1B1_m) - pi
    eq3 = asin(sA0B0_m) + asin(tA1B0_p) + asin(sA0B1_m) - asin(tA1B1_p) - pi
    eq4 = asin(sA0B0_m) + asin(tA1B0_m) + asin(sA0B1_m) - asin(tA1B1_m) - pi

    variables = [sA0B0_p, tA1B0_p, sA0B1_p, tA1B1_p,
                 sA0B0_m, tA1B0_m, sA0B1_m, tA1B1_m]

    jacobian = []
    for eq in [eq1, eq2, eq3, eq4]:
        row = [diff(eq, var) for var in variables]
        jacobian.append(row)

    jacobian_matrix = Matrix(jacobian)
    rank = jacobian_matrix.rank()
    print(f"\n[Symbolic Method] Jacobian matrix rank: {rank}")
    return rank


def check_numerical_independence(vars_dict):
    """Numerical method to check linear independence"""
    # Define symbolic variables
    vars_sym = symbols('sA0B0_p sA0B0_m tA1B0_p tA1B0_m sA0B1_p sA0B1_m tA1B1_p tA1B1_m')
    sA0B0_p, sA0B0_m, tA1B0_p, tA1B0_m, sA0B1_p, sA0B1_m, tA1B1_p, tA1B1_m = vars_sym

    # Define equations
    equations = [
        asin(sA0B0_p) + asin(tA1B0_p) + asin(sA0B1_p) - asin(tA1B1_p) - pi,
        asin(sA0B0_p) + asin(tA1B0_m) + asin(sA0B1_p) - asin(tA1B1_m) - pi,
        asin(sA0B0_m) + asin(tA1B0_p) + asin(sA0B1_m) - asin(tA1B1_p) - pi,
        asin(sA0B0_m) + asin(tA1B0_m) + asin(sA0B1_m) - asin(tA1B1_m) - pi
    ]

    # Build symbolic Jacobian
    jacobian_symbolic = []
    for eq in equations:
        row = [diff(eq, var) for var in vars_sym]
        jacobian_symbolic.append(row)

    # Substitute numerical values
    symbolic_to_value = {
        sA0B0_p: vars_dict['sA0B0plus'],
        sA0B0_m: vars_dict['sA0B0minus'],
        tA1B0_p: vars_dict['tA1B0plus'],
        tA1B0_m: vars_dict['tA1B0minus'],
        sA0B1_p: vars_dict['sA0B1plus'],
        sA0B1_m: vars_dict['sA0B1minus'],
        tA1B1_p: vars_dict['tA1B1plus'],
        tA1B1_m: vars_dict['tA1B1minus']
    }

    jacobian_numeric = []
    for row in jacobian_symbolic:
        jacobian_numeric.append([float(N(expr.subs(symbolic_to_value))) for expr in row])

    # Compute rank via SVD
    U, s, Vh = svd(jacobian_numeric)
    rank = np.sum(s > 1e-10)
    print(f"\n[Numerical Method] Jacobian matrix rank: {rank}")
    print("Singular values:", s)

    return rank, jacobian_numeric


def identify_independent_equations(vars_dict):
    """Identify exactly which equations are independent"""
    # Get numerical Jacobian matrix
    _, jacobian_numeric = check_numerical_independence(vars_dict)

    # Perform QR decomposition with column pivoting
    Q, R, piv = qr(np.array(jacobian_numeric).T, pivoting=True)

    # Determine independent equations (non-zero diagonal elements in R)
    tol = 1e-10
    rank = np.sum(np.abs(np.diag(R)) > tol)
    independent_indices = sorted(piv[:rank].tolist())

    print(f"\nIndependent equations (indices): {[i + 1 for i in independent_indices]}")

    # Analyze dependencies
    dependent_info = {}
    for i in range(4):
        if i not in independent_indices:
            # Solve for the linear combination
            A = np.array([jacobian_numeric[idx] for idx in independent_indices])
            b = np.array(jacobian_numeric[i])
            coeffs, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)

            # Verify the solution
            reconstructed = np.sum(coeffs.reshape(-1, 1) * A, axis=0)
            error = np.linalg.norm(reconstructed - b)

            if error < 1e-6:
                dependent_info[i] = {
                    'independent_eqs': independent_indices,
                    'coefficients': coeffs,
                    'error': error
                }

    # Print detailed results
    print("\nEquation dependency analysis:")
    eq_forms = [
        "arcsin(sA0B0+) + arcsin(tA1B0+) + arcsin(sA0B1+) - arcsin(tA1B1+) = π",
        "arcsin(sA0B0+) + arcsin(tA1B0-) + arcsin(sA0B1+) - arcsin(tA1B1-) = π",
        "arcsin(sA0B0-) + arcsin(tA1B0+) + arcsin(sA0B1-) - arcsin(tA1B1+) = π",
        "arcsin(sA0B0-) + arcsin(tA1B0-) + arcsin(sA0B1-) - arcsin(tA1B1-) = π"
    ]

    for i in range(4):
        if i in independent_indices:
            print(f"[Independent] Eq {i + 1}: {eq_forms[i]}")
        else:
            info = dependent_info.get(i, {})
            if info:
                print(f"[Dependent] Eq {i + 1}: {eq_forms[i]}")
                print("   Can be expressed as:")
                terms = []
                for j, c in zip(info['independent_eqs'], info['coefficients']):
                    terms.append(f"{c:.3f} × Eq {j + 1}")
                print("   + ".join(terms), f"(error: {info['error']:.2e})")
            else:
                print(f"[Unknown] Eq {i + 1}: {eq_forms[i]}")

    return independent_indices, dependent_info


# Main execution
if __name__ == "__main__":
    # Set parameters
    theta = np.pi / 6
    a0 = 0
    a1 = np.pi / 2

    # Calculate all parameters
    params = calculate_parameters(theta, a0, a1)
    vars_dict = compute_all_variables(params)

    # Print computed variables
    print("Computed variables:")
    for name, value in vars_dict.items():
        print(f"{name}: {value:.6f}")

    # Verify equations
    print("\nEquation verification:")
    results = verify_equations(vars_dict)
    for s, t, val, is_valid in results:
        print(f"s={s}, t={t}: value={val if isinstance(val, str) else f'{val:.6f}'}, valid={is_valid}")

    # Check independence
    print("\n=== Independence Analysis ===")
    symbolic_rank = check_linear_independence()
    numerical_rank, _ = check_numerical_independence(vars_dict)
    independent_eqs, _ = identify_independent_equations(vars_dict)

    print("\nFinal Summary:")
    print(f"Symbolic rank: {symbolic_rank}")
    print(f"Numerical rank: {numerical_rank}")
    print(f"Independent equations: {[i + 1 for i in independent_eqs]}")