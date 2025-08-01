import math
import numpy as np
from itertools import product
import random

# Pauli matrices
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

# Special angles for comparison
SPECIAL_ANGLES = {
    '0': 0,
    'π/12': math.pi / 12,
    'π/6': math.pi / 6,
    'π/4': math.pi / 4,
    'π/3': math.pi / 3,
    '5π/12': 5 * math.pi / 12,
    'π/2': math.pi / 2
}


def get_closest_special_angle(theta):
    """Return the name and value of the closest special angle"""
    closest_name = None
    closest_value = None
    min_diff = float('inf')

    for name, value in SPECIAL_ANGLES.items():
        diff = abs(theta - value)
        if diff < min_diff:
            min_diff = diff
            closest_name = name
            closest_value = value

    # Only consider it close if within 0.01 radians (~0.57 degrees)
    if min_diff < 0.01:
        return closest_name, closest_value
    return None, None


def calculate_parameters(theta, a0, a1, fixed_b0=None, fixed_b1=None):
    """Calculate measurement operators and expectation values"""
    psi = np.array([np.cos(theta), 0, 0, np.sin(theta)])
    sin2theta = np.sin(2 * theta)

    # Use fixed b0 and b1 if provided, otherwise calculate from current theta
    b0 = fixed_b0 if fixed_b0 is not None else np.arctan(sin2theta)
    b1 = fixed_b1 if fixed_b1 is not None else np.pi - np.arctan(sin2theta)

    mea_A0 = np.cos(a0) * Z + np.sin(a0) * X
    mea_A1 = np.cos(a1) * Z + np.sin(a1) * X
    mea_B0 = np.cos(b0) * Z + np.sin(b0) * X
    mea_B1 = - np.cos(b1) * Z - np.sin(b1) * X

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

    print("\nMeasurement angles:")
    print(f"a0: {a0:.6f}  a1: {a1:.6f}")
    print(f"b0: {b0:.6f}  b1: {b1:.6f}")

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


def decrease_theta(theta, iteration, min_theta=0.001):
    """Decrease theta with each iteration"""
    # Start with pi/24 and decrease by a factor each time
    new_theta = theta * (0.7 ** iteration)
    # Ensure we don't go below min_theta
    return max(new_theta, min_theta)


def calculate_angle_mod(a, theta, power):
    """计算 2arctan(tan(a/2) * tan(theta)^power) mod pi"""
    if math.isclose(a % math.pi, 0):
        return 0.0  # 处理tan(0)和tan(pi)的情况
    tan_half = math.tan(a / 2)
    tan_pow = math.tan(theta) ** power
    angle = 2 * math.atan(tan_half * tan_pow)
    return angle % math.pi


def check_angle_relations(a0, a1, b0, b1, theta, check_a1=True):
    """检查角度关系是否满足条件"""
    # 计算 [a0+], [a0-], [a1+], [a1-]
    a0_plus = calculate_angle_mod(a0, theta, 1)
    a0_minus = calculate_angle_mod(a0, theta, -1)
    a1_plus = calculate_angle_mod(a1, theta, 1)
    a1_minus = calculate_angle_mod(a1, theta, -1)

    # 按照要求格式输出各个角度
    print("角度计算结果:")
    print(f"a0+ = {a0_plus:.6f}")
    print(f"a0- = {a0_minus:.6f}")
    print(f"b0 = {b0:.6f}")
    print(f"a1+ = {a1_plus:.6f}")
    print(f"a1- = {a1_minus:.6f}")
    print(f"b1 = {b1:.6f}")

    # 检查条件1: [a0+]<=[a0-]且(如果需要)[a1+]<=[a1-]
    condition1 = a0_plus <= a0_minus
    if check_a1:
        condition1 = condition1 and (a1_plus <= a1_minus)

    # 检查条件2: 0<max([a0+],[a0-])<b0<min([a1+],[a1-])<b1<pi
    max_a0 = max(a0_plus, a0_minus)
    min_a1 = min(a1_plus, a1_minus)

    conditions2 = [
        (0 <= max_a0, f"0 <= max([a0+], [a0-]) ({max_a0:.6f})"),
        (max_a0 < b0, f"max([a0+], [a0-]) ({max_a0:.6f}) < b0 ({b0:.6f})"),
        (b0 < min_a1, f"b0 ({b0:.6f}) < min([a1+], [a1-]) ({min_a1:.6f})"),
        (min_a1 < b1, f"min([a1+], [a1-]) ({min_a1:.6f}) < b1 ({b1:.6f})"),
        (b1 <= math.pi, f"b1 ({b1:.6f}) <= pi ({math.pi:.6f})")
    ]

    print("\n条件1检查: [a0+] <= [a0-]" + (" 且 [a1+] <= [a1-]" if check_a1 else ""))
    print(f"[a0+] ({a0_plus:.6f}) <= [a0-] ({a0_minus:.6f}): {'满足' if a0_plus <= a0_minus else '不满足'}")
    if check_a1:
        print(f"[a1+] ({a1_plus:.6f}) <= [a1-] ({a1_minus:.6f}): {'满足' if a1_plus <= a1_minus else '不满足'}")
    print(f"条件1: {'满足' if condition1 else '不满足'}")

    print("\n条件2检查: 0 < max([a0+], [a0-]) < b0 < min([a1+], [a1-]) < b1 < pi")
    all_conditions2_met = True
    for condition, desc in conditions2:
        status = "满足" if condition else "不满足"
        print(f"{desc}: {status}")
        if not condition:
            all_conditions2_met = False

    print("\n结论:")
    print(f"条件1: {'满足' if condition1 else '不满足'}")
    print(f"条件2: {'满足' if all_conditions2_met else '不满足'}")
    print(f"所有条件: {'满足!' if (condition1 and all_conditions2_met) else '不满足!'}")

    return condition1 and all_conditions2_met


def run_simulation(initial_theta, a0, a1, iterations=10):
    """Run the simulation with theta decreasing"""
    theta = initial_theta
    min_theta = 0.001

    # Calculate fixed b0 and b1 based on sin(2*pi/6)
    fixed_b0 = np.arctan(np.sin(2 * math.pi/6))
    fixed_b1 = math.pi - fixed_b0

    first_iteration = True
    theta_values = []

    for i in range(iterations):
        print(f"\n=== Iteration {i + 1} ===")

        # Display theta value and closest special angle
        closest_name, closest_value = get_closest_special_angle(theta)
        if closest_name:
            print(f"Starting theta: {theta:.6f} (close to {closest_name} = {closest_value:.6f})")
        else:
            print(f"Starting theta: {theta:.6f} (not close to any standard angle)")
        print(f"Range: {min_theta:.6f} ≤ theta ≤ {initial_theta:.6f}")

        params = calculate_parameters(theta, a0, a1, fixed_b0, fixed_b1)

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

        # 检查角度关系
        print("\n检查角度关系:")
        check_a1 = first_iteration  # 只在第一次迭代检查[a1+]<=[a1-]
        all_conditions_met = check_angle_relations(a0, a1, params['b0'], params['b1'], theta, check_a1)
        first_iteration = False

        old_theta = theta
        theta = decrease_theta(theta, i)
        change = theta - old_theta  # This will be negative
        theta_values.append(theta)

        # Display new theta value and closest special angle
        closest_name, closest_value = get_closest_special_angle(theta)
        print(f"\nApplied change: {change:.6f}")
        if closest_name:
            print(f"New theta: {theta:.6f} (close to {closest_name} = {closest_value:.6f})")
        else:
            print(f"New theta: {theta:.6f} (not close to any standard angle)")
        print(f"Range: {min_theta:.6f} ≤ theta ≤ {initial_theta:.6f}")

    # Final check when theta is very small
    if theta > min_theta:
        print("\n=== Final iteration (theta near 0) ===")
        theta = min_theta
        print(f"Setting theta to minimum: {theta:.6f}")

        params = calculate_parameters(theta, a0, a1, fixed_b0, fixed_b1)

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

        # 检查角度关系
        print("\n检查角度关系:")
        check_angle_relations(a0, a1, params['b0'], params['b1'], theta, False)


# Set initial parameters
initial_theta = math.pi / 6  # Starting from pi/24
a0 = 0  # This remains fixed now
a1 = math.pi / 2

# Run the simulation with 10 iterations
run_simulation(initial_theta, a0, a1, iterations=10)