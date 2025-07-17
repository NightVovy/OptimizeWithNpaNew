import numpy as np
from sympy import symbols, cos, sin, sqrt, simplify


def calculate_terms_symbolically():
    # 定义符号变量
    A0, A1, B0, B1, E00, E01, E10, E11, s, t = symbols('A0 A1 B0 B1 E00 E01 E10 E11 s t')

    # 方块关联
    def new_corre(Exy, Ax, By, s):
        return (Exy + s * By) / (1 + s * Ax)

    # 计算8个分量
    terms = []

    # 1. ∂F/∂A0 * A0
    term1 = A0 * (
            - (s * (E00 + s * B0)) / ((1 + s * A0) ** 2 * sqrt(1 - new_corre(E00, A0, B0, s) ** 2))
            + (s * (E01 + s * B1)) / ((1 + s * A0) ** 2 * sqrt(1 - new_corre(E01, A0, B1, s) ** 2))
    )
    terms.append(term1)

    # 2. ∂F/∂A1 * A1
    term2 = A1 * (
            - (t * (E10 + t * B0)) / ((1 + t * A1) ** 2 * sqrt(1 - new_corre(E10, A1, B0, t) ** 2))
            - (t * (E11 + t * B1)) / ((1 + t * A1) ** 2 * sqrt(1 - new_corre(E11, A1, B1, t) ** 2))
    )
    terms.append(term2)

    # 3. ∂F/∂B0 * B0
    term3 = B0 * (
            (s * 1) / ((1 + s * A0) * sqrt(1 - new_corre(E00, A0, B0, s) ** 2))
            + (t * 1) / ((1 + t * A1) * sqrt(1 - new_corre(E10, A1, B0, t) ** 2))
    )
    terms.append(term3)

    # 4. ∂F/∂B1 * B1
    term4 = B1 * (
            - (s * 1) / ((1 + s * A0) * sqrt(1 - new_corre(E01, A0, B1, s) ** 2))
            + (t * 1) / ((1 + t * A1) * sqrt(1 - new_corre(E11, A1, B1, t) ** 2))
    )
    terms.append(term4)

    # 5. ∂F/∂E00 * E00
    term5 = E00 * 1 / ((1 + s * A0) * sqrt(1 - new_corre(E00, A0, B0, s) ** 2))
    terms.append(term5)

    # 6. ∂F/∂E01 * E01
    term6 = E01 * -1 / ((1 + s * A0) * sqrt(1 - new_corre(E01, A0, B1, s) ** 2))
    terms.append(term6)

    # 7. ∂F/∂E10 * E10
    term7 = E10 * 1 / ((1 + t * A1) * sqrt(1 - new_corre(E10, A1, B0, t) ** 2))
    terms.append(term7)

    # 8. ∂F/∂E11 * E11
    term8 = E11 * 1 / ((1 + t * A1) * sqrt(1 - new_corre(E11, A1, B1, t) ** 2))
    terms.append(term8)

    # 显示符号形式的terms
    print("符号形式的8个terms:")
    for i, term in enumerate(terms, 1):
        print(f"Term {i}: {term}")
    print("\n")

    # 定义替换规则
    theta, a0, a1, b0, b1 = symbols('theta a0 a1 b0 b1')
    substitutions = {
        A0: cos(2 * theta) * cos(a0),
        A1: cos(2 * theta) * cos(a1),
        B0: cos(2 * theta) * cos(b0),
        B1: cos(2 * theta) * cos(b1),
        E00: cos(a0) * cos(b0) + sin(2 * theta) * sin(a0) * sin(b0),
        E01: cos(a0) * cos(b1) + sin(2 * theta) * sin(a0) * sin(b1),
        E10: cos(a1) * cos(b0) + sin(2 * theta) * sin(a1) * sin(b0),
        E11: cos(a1) * cos(b1) + sin(2 * theta) * sin(a1) * sin(b1)
    }

    # 应用替换
    substituted_terms = [term.subs(substitutions) for term in terms]

    # 显示替换后的terms
    print("代入给定表达式后的8个terms:")
    for i, term in enumerate(substituted_terms, 1):
        print(f"Term {i}: {term}")

    return terms, substituted_terms


# 执行函数
symbolic_terms, substituted_terms = calculate_terms_symbolically()