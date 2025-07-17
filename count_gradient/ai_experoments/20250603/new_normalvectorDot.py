import numpy as np

def calculate_normal_vector_dot_product(A0, A1, B0, B1, E00, E01, E10, E11, s, t):

    # 方块关联
    def new_corre(Exy, Ax, By, s):
        return (Exy + s * By)/(1 + s * Ax)

    # 计算8个分量
    terms = []

    # 1. ∂F/∂A0 * A0
    term1 =  (
        - (s * (E00 + s * B0) * A0)/((1 + s * A0)** 2 * np.sqrt(1 - new_corre(E00,A0,B0,s)** 2))
        + (s * (E01 + s * B1) * A0)/((1 + s * A0)** 2 * np.sqrt(1 - new_corre(E01,A0,B1,s)** 2))
    )
    terms.append(term1)

    # 2. ∂F/∂A1 * A1
    term2 =  (
        - (t * (E10 + t * B0) * A1)/((1 + t * A1)** 2 * np.sqrt(1 - new_corre(E10,A1,B0,t)** 2))
        - (t * (E11 + t * B1) * A1)/((1 + t * A1)** 2 * np.sqrt(1 - new_corre(E11,A1,B1,t)** 2))
    )
    terms.append(term2)

    # 3. ∂F/∂B0 * B0
    term3 =  (
        (s * B0)/((1 + s * A0) * np.sqrt(1 - new_corre(E00,A0,B0,s)** 2))
        + (t * B0)/((1 + t * A1) * np.sqrt(1 - new_corre(E10,A1,B0,t)** 2))
    )
    terms.append(term3)

    # 4. ∂F/∂B1 * B1
    term4 =  (
            - (s * B1) / ((1 + s * A0) * np.sqrt(1 - new_corre(E01, A0, B1, s) ** 2))
            + (t * B1) / ((1 + t * A1) * np.sqrt(1 - new_corre(E11, A1, B1, t) ** 2))
    )
    terms.append(term4)

    # 5. ∂F/∂E00 * E00
    term5 = E00 / ((1 + s * A0) * np.sqrt(1 - new_corre(E00,A0,B0,s)** 2))
    terms.append(term5)

    # 6. ∂F/∂E01 * E01
    term6 = - E01 / ((1 + s * A0) * np.sqrt(1 - new_corre(E01, A0, B1, s) ** 2))
    terms.append(term6)

    # 7. ∂F/∂E10 * E10
    term7 = E10 / ((1 + t * A1) * np.sqrt(1 - new_corre(E10,A1,B0,t)** 2))
    terms.append(term7)

    # 8. ∂F/∂E11 * E11
    term8 = E11 / ((1 + t * A1) * np.sqrt(1 - new_corre(E11, A1, B1, t) ** 2))
    terms.append(term8)

    total = sum(terms)

    return terms, total









