import math
import numpy as np
from itertools import product
import random

# Pauli matrices
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)


def calculate_parameters(theta, a0, a1, b0, b1):
    """Calculate measurement operators and expectation values with fixed b0 and b1"""
    psi = np.array([np.cos(theta), 0, 0, np.sin(theta)])

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

        terms = [sA0B0, tA1B0, -sA0B1, tA1B1]
        if any(abs(term) > 1 for term in terms):
            results.append((s, t, "Undefined (arcsin domain error)", sA0B0, tA1B0, sA0B1, tA1B1))
            continue

        value = (math.asin(sA0B0) + math.asin(tA1B0) -
                 math.asin(sA0B1) + math.asin(tA1B1))
        is_valid = math.isclose(value, math.pi, rel_tol=1e-9)

        results.append((s, t, value, is_valid, sA0B0, tA1B0, sA0B1, tA1B1))
    return results


def calculate_angle_mod(a, theta, power):
    """计算 2arctan(tan(a/2) * tan(theta)^power) mod pi"""
    if math.isclose(a % math.pi, 0):
        return 0.0  # 处理tan(0)和tan(pi)的情况
    tan_half = math.tan(a / 2)
    tan_pow = math.tan(theta) ** power
    angle = 2 * math.atan(tan_half * tan_pow)
    return angle  # 直接返回角度值，不再取模
    # return angle % math.pi


def check_angle_relations(a0, a1, b0, b1, theta):
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

    # 检查条件1: [a0+]>=[a0-] 和 [a1+]<=[a1-]   有修改
    condition1 = a0_plus >= a0_minus and a1_plus <= a1_minus

    # 检查条件2: 各个角度的大小关系
    conditions2 = [
        # (0 <= a0_plus, f"0 <= [a0+] ({a0_plus:.6f})"),
        (a0_plus >= a0_minus, f"[a0+] ({a0_plus:.6f}) <= [a0-] ({a0_minus:.6f})"),
        (a0_plus <= b0, f"[a0-] ({a0_minus:.6f}) <= b0 ({b0:.6f})"),
        (b0 <= a1_plus, f"b0 ({b0:.6f}) <= [a1+] ({a1_plus:.6f})"),
        (a1_plus <= a1_minus, f"[a1+] ({a1_plus:.6f}) <= [a1-] ({a1_minus:.6f})"),
        (a1_minus <= b1, f"[a1-] ({a1_minus:.6f}) <= b1 ({b1:.6f})")
        # (b1 <= math.pi, f"b1 ({b1:.6f}) <= pi ({math.pi:.6f})")
    ]

    print("\n条件1检查: [a0+] >= [a0-] 且 [a1+] <= [a1-]")
    print(f"[a0+] ({a0_plus:.6f}) >= [a0-] ({a0_minus:.6f}): {'满足' if a0_plus >= a0_minus else '不满足'}")
    print(f"[a1+] ({a1_plus:.6f}) <= [a1-] ({a1_minus:.6f}): {'满足' if a1_plus <= a1_minus else '不满足'}")
    print(f"条件1: {'满足' if condition1 else '不满足'}")

    print("\n条件2检查:  [a0-] <= [a0+] <= b0 <= [a1+] <= [a1-] <= b1 ")
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


def run_simulation(theta):
    """Run the simulation with fixed angles"""
    sin2theta = np.sin(2 * theta)
    a0 = -np.arctan(sin2theta)
    a1 = np.arctan(sin2theta)
    b0 = 0
    b1 = np.pi / 2

    print(f"=== Initial Parameters ===")
    print(f"theta: {theta:.6f}")
    print(f"a0: {a0:.6f} (-arctan(sin(2θ)))")
    print(f"a1: {a1:.6f} (arctan(sin(2θ)))")
    print(f"b0: {b0:.6f}")
    print(f"b1: {b1:.6f} (π/2)")

    params = calculate_parameters(theta, a0, a1, b0, b1)

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
    check_angle_relations(a0, a1, b0, b1, theta)


# Set parameters
theta = np.pi / 6

# Run the simulation
run_simulation(theta)