from sympy import symbols, sqrt, Matrix, simplify
import numpy as np


def calculate_normalized_normal_vector(symbolic=True, param_values=None):
    """
    计算归一化的法向量

    参数:
        symbolic: 是否返回符号表达式 (True) 或数值结果 (False)
        param_values: 当 symbolic=False 时，需要提供的参数值字典，例如:
            {'A0': 1.0, 'A1': 0.5, 'B0': 0.2, 'B1': 0.3,
             'E00': 0.1, 'E01': 0.4, 'E10': 0.2, 'E11': 0.5,
             's': 0.8, 't': 0.6}

    返回:
        归一化的法向量 (符号表达式或数值数组)
    """
    # 定义符号变量
    A0, A1, B0, B1, E00, E01, E10, E11, s, t = symbols('A0 A1 B0 B1 E00 E01 E10 E11 s t')

    # 方块关联函数
    def new_corre(Exy, Ax, By, s):
        return (Exy + s * By) / (1 + s * Ax)

    # 计算8个分量
    terms = []

    # 1. ∂F/∂A0
    term1 = (
            - (s * (E00 + s * B0)) / ((1 + s * A0) ** 2 * sqrt(1 - new_corre(E00, A0, B0, s) ** 2))
            + (s * (E01 + s * B1)) / ((1 + s * A0) ** 2 * sqrt(1 - new_corre(E01, A0, B1, s) ** 2))
    )
    terms.append(term1)

    # 2. ∂F/∂A1
    term2 = (
            - (t * (E10 + t * B0)) / ((1 + t * A1) ** 2 * sqrt(1 - new_corre(E10, A1, B0, t) ** 2))
            - (t * (E11 + t * B1)) / ((1 + t * A1) ** 2 * sqrt(1 - new_corre(E11, A1, B1, t) ** 2))
    )
    terms.append(term2)

    # 3. ∂F/∂B0
    term3 = (
            (s * 1) / ((1 + s * A0) * sqrt(1 - new_corre(E00, A0, B0, s) ** 2))
            + (t * 1) / ((1 + t * A1) * sqrt(1 - new_corre(E10, A1, B0, t) ** 2))
    )
    terms.append(term3)

    # 4. ∂F/∂B1
    term4 = (
            - (s * 1) / ((1 + s * A0) * sqrt(1 - new_corre(E01, A0, B1, s) ** 2))
            + (t * 1) / ((1 + t * A1) * sqrt(1 - new_corre(E11, A1, B1, t) ** 2))
    )
    terms.append(term4)

    # 5. ∂F/∂E00
    term5 = 1 / ((1 + s * A0) * sqrt(1 - new_corre(E00, A0, B0, s) ** 2))
    terms.append(term5)

    # 6. ∂F/∂E01
    term6 = -1 / ((1 + s * A0) * sqrt(1 - new_corre(E01, A0, B1, s) ** 2))
    terms.append(term6)

    # 7. ∂F/∂E10
    term7 = 1 / ((1 + t * A1) * sqrt(1 - new_corre(E10, A1, B0, t) ** 2))
    terms.append(term7)

    # 8. ∂F/∂E11
    term8 = 1 / ((1 + t * A1) * sqrt(1 - new_corre(E11, A1, B1, t) ** 2))
    terms.append(term8)

    if symbolic:
        # 符号计算归一化
        n = Matrix(terms)
        norm = sqrt(sum([simplify(term ** 2) for term in terms]))
        n_normalized = n / norm
        return n_normalized
    else:
        # 数值计算归一化
        if param_values is None:
            raise ValueError("当 symbolic=False 时，必须提供 param_values")

        # 替换参数值
        subs_dict = {A0: param_values['A0'], A1: param_values['A1'],
                     B0: param_values['B0'], B1: param_values['B1'],
                     E00: param_values['E00'], E01: param_values['E01'],
                     E10: param_values['E10'], E11: param_values['E11'],
                     s: param_values['s'], t: param_values['t']}

        # 计算数值向量
        numerical_terms = [term.subs(subs_dict).evalf() for term in terms]
        n_array = np.array(numerical_terms, dtype=np.float64)

        # 归一化
        norm = np.linalg.norm(n_array)
        n_normalized = n_array / norm
        return n_normalized


# 示例用法
if __name__ == "__main__":
    # 1. 符号计算
    symbolic_result = calculate_normalized_normal_vector(symbolic=True)
    print("符号表达式归一化结果:")
    for i, term in enumerate(symbolic_result):
        print(f"n_{i + 1} = {term}")

    # 2. 数值计算
    params = {'A0': 0.707107, 'A1': 0, 'B0': 0.577350, 'B1': 0.577350,
              'E00': 0.816497, 'E01': 0.816497, 'E10': 0.408248, 'E11': 0.408248,
              's': 1, 't': 1}

    numerical_result = calculate_normalized_normal_vector(symbolic=False, param_values=params)
    print("\n数值归一化结果:")
    print(numerical_result)