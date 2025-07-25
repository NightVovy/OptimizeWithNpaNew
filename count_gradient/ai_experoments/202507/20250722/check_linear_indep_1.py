import math
import numpy as np
from sympy import symbols, Matrix, asin, pi, diff, simplify, N
from scipy.linalg import svd

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
    mea_B1 = np.cos(b1) * Z - np.sin(b1) * X  # notice

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
    """Compute all 8 variables"""
    A0, A1, B0, B1 = params['A0'], params['A1'], params['B0'], params['B1']
    E00, E01, E10, E11 = params['E00'], params['E01'], params['E10'], params['E11']

    # Calculate all 8 variables
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
    """Verify all 4 equations"""
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


def check_linear_independence(vars_dict):
    """Check linear independence using Jacobian matrix"""
    # Define symbolic variables
    sA0B0_p, sA0B0_m = symbols('sA0B0_p sA0B0_m')
    tA1B0_p, tA1B0_m = symbols('tA1B0_p tA1B0_m')
    sA0B1_p, sA0B1_m = symbols('sA0B1_p sA0B1_m')
    tA1B1_p, tA1B1_m = symbols('tA1B1_p tA1B1_m')

    # Define the 4 equations
    eq1 = asin(sA0B0_p) + asin(tA1B0_p) + asin(sA0B1_p) - asin(tA1B1_p) - pi
    eq2 = asin(sA0B0_p) + asin(tA1B0_m) + asin(sA0B1_p) - asin(tA1B1_m) - pi
    eq3 = asin(sA0B0_m) + asin(tA1B0_p) + asin(sA0B1_m) - asin(tA1B1_p) - pi
    eq4 = asin(sA0B0_m) + asin(tA1B0_m) + asin(sA0B1_m) - asin(tA1B1_m) - pi

    # Variables in the order we want to differentiate
    variables = [sA0B0_p, tA1B0_p, sA0B1_p, tA1B1_p,
                 sA0B0_m, tA1B0_m, sA0B1_m, tA1B1_m]

    # Build Jacobian matrix
    jacobian = []
    for eq in [eq1, eq2, eq3, eq4]:
        row = [diff(eq, var) for var in variables]
        jacobian.append(row)

    jacobian_matrix = Matrix(jacobian)
    rank = jacobian_matrix.rank()

    print(f"\nJacobian matrix rank: {rank}")
    print(f"Number of linearly independent equations: {rank}")

    return rank


def check_numerical_independence(vars_dict):
    """数值方法验证方程独立性（基于具体变量值）"""
    # 定义符号变量
    sA0B0_p, tA1B0_p, sA0B1_p, tA1B1_p = symbols('sA0B0_p tA1B0_p sA0B1_p tA1B1_p')
    sA0B0_m, tA1B0_m, sA0B1_m, tA1B1_m = symbols('sA0B0_m tA1B0_m sA0B1_m tA1B1_m')

    # 定义符号方程
    eq1 = asin(sA0B0_p) + asin(tA1B0_p) + asin(sA0B1_p) - asin(tA1B1_p) - pi
    eq2 = asin(sA0B0_p) + asin(tA1B0_m) + asin(sA0B1_p) - asin(tA1B1_m) - pi
    eq3 = asin(sA0B0_m) + asin(tA1B0_p) + asin(sA0B1_m) - asin(tA1B1_p) - pi
    eq4 = asin(sA0B0_m) + asin(tA1B0_m) + asin(sA0B1_m) - asin(tA1B1_m) - pi

    # 变量列表
    variables = [sA0B0_p, tA1B0_p, sA0B1_p, tA1B1_p,
                 sA0B0_m, tA1B0_m, sA0B1_m, tA1B1_m]

    # 构建符号到数值的映射
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

    # 计算数值Jacobian矩阵
    jacobian = []
    for eq in [eq1, eq2, eq3, eq4]:
        row = []
        for var in variables:
            # 计算符号导数并代入数值
            derivative_expr = diff(eq, var)
            derivative_val = derivative_expr.subs(symbolic_to_value)
            row.append(float(N(derivative_val)))  # 转换为浮点数
        jacobian.append(row)

    # 使用SVD计算数值秩
    U, s, Vh = svd(jacobian)
    rank = np.sum(s > 1e-10)  # 基于奇异值阈值判断秩

    print(f"[数值方法] Jacobian矩阵秩: {rank}")
    print(f"奇异值: {s}")
    return rank


# Main execution
theta = np.pi / 6
a0 = 0
a1 = np.pi / 2

params = calculate_parameters(theta, a0, a1)
vars_dict = compute_all_variables(params)

print("Computed variables:")
for name, value in vars_dict.items():
    print(f"{name}: {value:.6f}")

print("\nEquation verification:")
results = verify_equations(vars_dict)
for s, t, val, is_valid in results:
    print(f"\nEquation s={s}, t={t}:")
    print(f"Value: {val if isinstance(val, str) else f'{val:.6f}'}")
    print(f"Valid (equals pi): {is_valid}")

# Check linear independence
rank = check_linear_independence(vars_dict)
numerical_rank = check_numerical_independence(vars_dict)  # 数值方法