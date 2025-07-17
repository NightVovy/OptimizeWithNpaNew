import numpy as np


def calculate_normal_vector_dot_product(A0, A1, B0, B1, E00, E01, E10, E11, s=1, t=1):
    """
    计算法向量点积 ∇F·E 的8个分量及总和

    参数:
        A0, A1, B0, B1, E00, E01, E10, E11: 变量值
        s, t: 常数(默认为1)

    返回:
        terms: 8个分量结果的列表
        total: 总和
    """

    # 计算各项分母和平方根部分
    def sqrt_term(x, y, z):
        arg = (x + y * z) / (1 + y * z)
        return (1 + y * z) * np.sqrt(1 - arg ** 2)

    # 计算8个分量
    terms = []

    # 1. ∂F/∂A0 * A0
    term1 = (-s * (E00 + s * B0) / ((1 + s * A0) ** 2 * sqrt_term(E00, s, B0))
             + s * (E01 + s * B1) / ((1 + s * A0) ** 2 * sqrt_term(E01, s, B1))) * A0
    terms.append(term1)

    # 2. ∂F/∂A1 * A1
    term2 = (-t * (E10 + t * B0) / ((1 + t * A1) ** 2 * sqrt_term(E10, t, B0))
             - t * (E11 + t * B1) / ((1 + t * A1) ** 2 * sqrt_term(E11, t, B1))) * A1
    terms.append(term2)

    # 3. ∂F/∂B0 * B0
    term3 = (s / ((1 + s * A0) * sqrt_term(E00, s, B0))
             + t / ((1 + t * A1) * sqrt_term(E10, t, B0))) * B0
    terms.append(term3)

    # 4. ∂F/∂B1 * B1
    term4 = (t / ((1 + t * A1) * sqrt_term(E11, t, B1))
             - s / ((1 + s * A0) * sqrt_term(E01, s, B1))) * B1
    terms.append(term4)

    # 5. ∂F/∂E00 * E00
    term5 = (1 / ((1 + s * A0) * sqrt_term(E00, s, B0))) * E00
    terms.append(term5)

    # 6. ∂F/∂E01 * E01
    term6 = (-1 / ((1 + s * A0) * sqrt_term(E01, s, B1))) * E01
    terms.append(term6)

    # 7. ∂F/∂E10 * E10
    term7 = (1 / ((1 + t * A1) * sqrt_term(E10, t, B0))) * E10
    terms.append(term7)

    # 8. ∂F/∂E11 * E11
    term8 = (1 / ((1 + t * A1) * sqrt_term(E11, t, B1))) * E11
    terms.append(term8)

    total = sum(terms)

    return terms, total


# 示例使用
if __name__ == "__main__":
    # 输入变量值 (这里用示例值，可以修改为实际值)
    A0, A1 = 0.1, 0.2
    B0, B1 = 0.3, 0.4
    E00, E01, E10, E11 = 0.5, 0.6, 0.7, 0.8
    s, t = 1, 1  # 可以改为-1如果需要

    # 计算并输出结果
    terms, total = calculate_normal_vector_dot_product(A0, A1, B0, B1, E00, E01, E10, E11, s, t)

    print("法向量点积的8个分量:")
    for i, term in enumerate(terms, 1):
        print(f"项 {i}: {term:.6f}")

    print(f"\n总和: {total:.6f}")