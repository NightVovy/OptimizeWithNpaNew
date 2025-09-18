import numpy as np
import sympy as sp
from sympy import pi

# 定义符号变量
theta, a0, a1, b0, b1 = sp.symbols('theta a0 a1 b0 b1', real=True)

# 定义基向量 |00>, |01>, |10>, |11>
basis = ['00', '01', '10', '11']


def ket_to_vector(ket_coeffs):
    """将ket表达式转换为4维向量"""
    vector = np.zeros(4, dtype=object)
    for i, state in enumerate(basis):
        if state in ket_coeffs:
            vector[i] = ket_coeffs[state]
        else:
            vector[i] = 0
    return vector


def inner_product(vec1, vec2):
    """计算两个向量的内积"""
    result = 0
    for i in range(4):
        result += sp.conjugate(vec1[i]) * vec2[i]
    return sp.simplify(result)


# 定义 |psi> = cosθ|00> + sinθ|11>
psi_coeffs = {'00': sp.cos(theta), '11': sp.sin(theta)}
psi_vector = ket_to_vector(psi_coeffs)

# 定义 |vec> = sinθ|00> - cosθ|11>
vec_coeffs = {'00': sp.sin(theta), '11': -sp.cos(theta)}
vec_vector = ket_to_vector(vec_coeffs)

# 定义 |vec> = |01>
# vec_coeffs = {'01': 1}  # 只有 |01> 的系数为 1，其他为 0
# vec_vector = ket_to_vector(vec_coeffs)

# 定义 |vec> = |10>
# vec_coeffs = {'10': 1}  # 只有 |01> 的系数为 1，其他为 0
# vec_vector = ket_to_vector(vec_coeffs)


# 定义Pauli矩阵作用
def Z_action(state):
    """Z|0> = |0>, Z|1> = -|1>"""
    if state == '0':
        return {'0': 1}
    else:
        return {'1': -1}


def X_action(state):
    """X|0> = |1>, X|1> = |0>"""
    if state == '0':
        return {'1': 1}
    else:
        return {'0': 1}


def apply_operator_A(operator, state, angle):
    """应用A算子: cos(angle)*Z + sin(angle)*X"""
    result = {}

    # Z部分
    z_result = Z_action(state)
    for key, value in z_result.items():
        result[key] = result.get(key, 0) + sp.cos(angle) * value

    # X部分
    x_result = X_action(state)
    for key, value in x_result.items():
        result[key] = result.get(key, 0) + sp.sin(angle) * value

    return result


def apply_operator_B(operator, state, angle):
    """应用B算子: cos(angle)*Z + sin(angle)*X"""
    return apply_operator_A(operator, state, angle)


def tensor_product_operator_A_I(operator, ket_coeffs, angle):
    """计算 (A⊗I)|ket>"""
    result = {}

    for state, coeff in ket_coeffs.items():
        if coeff == 0:
            continue

        qubit1, qubit2 = state[0], state[1]
        A_result = apply_operator_A(None, qubit1, angle)

        for A_state, A_value in A_result.items():
            new_state = A_state + qubit2
            result[new_state] = result.get(new_state, 0) + coeff * A_value

    return result


def tensor_product_operator_I_B(operator, ket_coeffs, angle):
    """计算 (I⊗B)|ket>"""
    result = {}

    for state, coeff in ket_coeffs.items():
        if coeff == 0:
            continue

        qubit1, qubit2 = state[0], state[1]
        B_result = apply_operator_B(None, qubit2, angle)

        for B_state, B_value in B_result.items():
            new_state = qubit1 + B_state
            result[new_state] = result.get(new_state, 0) + coeff * B_value

    return result


def tensor_product_operator_A_B(operator, ket_coeffs, angle_A, angle_B):
    """计算 (A⊗B)|ket>"""
    result = {}

    for state, coeff in ket_coeffs.items():
        if coeff == 0:
            continue

        qubit1, qubit2 = state[0], state[1]
        A_result = apply_operator_A(None, qubit1, angle_A)
        B_result = apply_operator_B(None, qubit2, angle_B)

        for A_state, A_value in A_result.items():
            for B_state, B_value in B_result.items():
                new_state = A_state + B_state
                result[new_state] = result.get(new_state, 0) + coeff * A_value * B_value

    return result


# 计算 M|psi> 的所有分量
def compute_M_psi(psi_coeffs):
    """计算 M|psi> 的所有8个分量"""
    M_psi_components = []

    # 1. (A0⊗I)|psi>
    comp1 = tensor_product_operator_A_I(None, psi_coeffs, a0)
    M_psi_components.append(comp1)

    # 2. (A1⊗I)|psi>
    comp2 = tensor_product_operator_A_I(None, psi_coeffs, a1)
    M_psi_components.append(comp2)

    # 3. (I⊗B0)|psi>
    comp3 = tensor_product_operator_I_B(None, psi_coeffs, b0)
    M_psi_components.append(comp3)

    # 4. (I⊗B1)|psi>
    comp4 = tensor_product_operator_I_B(None, psi_coeffs, b1)
    M_psi_components.append(comp4)

    # 5. (A0⊗B0)|psi>
    comp5 = tensor_product_operator_A_B(None, psi_coeffs, a0, b0)
    M_psi_components.append(comp5)

    # 6. (A0⊗B1)|psi>
    comp6 = tensor_product_operator_A_B(None, psi_coeffs, a0, b1)
    M_psi_components.append(comp6)

    # 7. (A1⊗B0)|psi>
    comp7 = tensor_product_operator_A_B(None, psi_coeffs, a1, b0)
    M_psi_components.append(comp7)

    # 8. (A1⊗B1)|psi>
    comp8 = tensor_product_operator_A_B(None, psi_coeffs, a1, b1)
    M_psi_components.append(comp8)

    return M_psi_components


def compute_inner_product(vec_coeffs, M_psi_component):
    """计算 <vec|(M分量)|psi>"""
    # 将M分量转换为向量形式
    component_vector = ket_to_vector(M_psi_component)

    # 计算内积 <vec|(M分量)|psi>
    result = inner_product(vec_vector, component_vector)
    return sp.simplify(result)


def substitute_values(expression, theta_val, a0_val, a1_val, b0_val, b1_val):
    """代入具体的数值"""
    return expression.subs({
        theta: theta_val,
        a0: a0_val,
        a1: a1_val,
        b0: b0_val,
        b1: b1_val
    }).evalf()


def get_vec_description(vec_coeffs):
    """根据vec_coeffs自动生成描述字符串"""
    terms = []
    for state, coeff in vec_coeffs.items():
        if coeff == 1:
            terms.append(f"|{state}>")
        elif coeff == -1:
            terms.append(f"-|{state}>")
        else:
            terms.append(f"{coeff}|{state}>")

    if not terms:
        return "|0>"

    # 连接各项，处理首项的符号
    description = terms[0]
    for term in terms[1:]:
        if term.startswith('-'):
            description += f" {term}"
        else:
            description += f" + {term}"

    return description

# 主计算函数
def main(use_numerical_values=False):
    vec_description = get_vec_description(vec_coeffs)
    print("=" * 60)
    print(f"当前 |vec> = {vec_description}")
    print("=" * 60)

    # 计算 M|psi> 的所有分量
    M_psi_components = compute_M_psi(psi_coeffs)

    print("M|psi> 的8个分量（符号形式）:")
    print("-" * 50)
    for i, component in enumerate(M_psi_components):
        print(f"分量 {i + 1}:")
        for state, coeff in component.items():
            if coeff != 0:
                print(f"  |{state}>: {coeff}")
        print()

    print("计算 <vec|M|psi>（符号形式）:")
    print("-" * 50)

    symbolic_results = []
    for i, component in enumerate(M_psi_components):
        inner_prod = compute_inner_product(vec_coeffs, component)
        symbolic_results.append(inner_prod)
        print(f"<vec|M{i + 1}|psi> = {inner_prod}")

    if use_numerical_values:
        print("\n" + "=" * 60)
        print("代入数值计算:")
        print(f"θ = π/6 ≈ {pi.evalf() / 6:.6f}")
        print(f"a0 = {0}")
        print(f"a1 = {1.42744876}")
        print(f"b0 = {0.71372438}")
        print(f"b1 = {2.28452071}")
        print("=" * 60)

        # 定义数值
        theta_val = pi / 6
        a0_val = 0
        a1_val = 1.42744876
        b0_val = 0.71372438
        b1_val = 2.28452071

        numerical_results = []
        for i, expr in enumerate(symbolic_results):
            numerical_value = substitute_values(expr, theta_val, a0_val, a1_val, b0_val, b1_val)
            numerical_results.append(numerical_value)
            print(f"<vec|M{i + 1}|psi> = {numerical_value:.8f}")

        return symbolic_results, numerical_results

    return symbolic_results, None


if __name__ == "__main__":
    # 首先输出符号形式的结果
    symbolic_results, _ = main(use_numerical_values=False)

    print("\n" + "=" * 60)
    print("现在进行数值计算...")
    print("=" * 60)

    # 然后输出数值结果
    symbolic_results, numerical_results = main(use_numerical_values=True)

    # 输出总结
    print("\n" + "=" * 60)
    print("结果总结:")
    print("=" * 60)
    print("符号结果:")
    for i, expr in enumerate(symbolic_results):
        print(f"<vec|M{i + 1}|psi> = {expr}")

    print("\n数值结果:")
    for i, val in enumerate(numerical_results):
        print(f"<vec|M{i + 1}|psi> = {val:.8f}")