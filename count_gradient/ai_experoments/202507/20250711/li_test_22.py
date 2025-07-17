import numpy as np
import math

# 定义Pauli矩阵
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])


def calculate_correlations(theta, a0, a1, b0=None, b1=None):
    """
    计算量子关联算子
    """
    psi = np.array([np.cos(theta), 0, 0, np.sin(theta)])

    calculated_b = None
    if b0 is None or b1 is None:
        sin2theta = np.sin(2 * theta)
        b0 = np.arctan(sin2theta)
        b1 = np.arctan(sin2theta)
        calculated_b = (b0, b1)

    alpha = 2 / math.sqrt(1 + 2 * (math.tan(2 * theta)) ** 2)
    sqrt_value = math.sqrt(8 + 2 * (alpha ** 2))

    A0 = np.cos(a0) * Z + np.sin(a0) * X
    A1 = np.cos(a1) * Z + np.sin(a1) * X
    B0 = np.cos(b0) * Z + np.sin(b0) * X
    B1 = np.cos(b1) * Z - np.sin(b1) * X

    def tensor_op(op1, op2):
        return np.kron(op1, op2)

    def expectation_value(op):
        return np.dot(psi.conj(), np.dot(op, psi)).real

    return (
        expectation_value(tensor_op(A0, I)),  # EA0
        expectation_value(tensor_op(A1, I)),  # EA1
        expectation_value(tensor_op(I, B0)),  # EB0
        expectation_value(tensor_op(I, B1)),  # EB1
        expectation_value(tensor_op(A0, B0)),  # EA0B0
        expectation_value(tensor_op(A0, B1)),  # EA0B1
        expectation_value(tensor_op(A1, B0)),  # EA1B0
        expectation_value(tensor_op(A1, B1)),  # EA1B1
        (b0, b1),
        alpha,
        sqrt_value
    )


def calculate_new_formula(EA0, EA1, EB0, EB1, EA0B0, EA0B1, EA1B0, EA1B1, s=1, t=1):
    """
    计算新公式的值
    """
    term1 = math.asin((EA0B0 + s * EB0) / (1 + s * EA0))
    term2 = math.asin((EA0B1 + s * EB1) / (1 + s * EA0))
    term3 = math.asin((EA1B0 + t * EB0) / (1 + t * EA1))
    term4 = math.asin((EA1B1 + t * EB1) / (1 + t * EA1))

    result = term1 + term2 + term3 - term4
    return result, term1, term2, term3, term4


def check_linear_independence(results):
    """
    检查四种result的线性独立性
    参数results是包含4个result值的列表
    返回矩阵的秩和线性无关的数量
    """
    # 我们需要构建一个4×4矩阵，其中每行是不同(s,t)组合的result展开
    # 但由于result是单个值，我们需要考虑它们作为向量空间中的向量

    # 将results转换为列向量
    matrix = np.array(results).reshape(-1, 1)

    # 计算矩阵的秩
    rank = np.linalg.matrix_rank(matrix)

    return rank


# 输入参数
theta = np.pi / 6
a0 = 0
a1 = np.pi / 2

# 计算关联算子
EA0, EA1, EB0, EB1, EA0B0, EA0B1, EA1B0, EA1B1, (b0, b1), alpha, sqrt_value = calculate_correlations(theta, a0, a1)

# 定义所有s,t组合
st_combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
results = []

# 计算所有组合的结果
results = []
for s, t in st_combinations:
    formula_result, term1, term2, term3, term4 = calculate_new_formula(
        EA0, EA1, EB0, EB1, EA0B0, EA0B1, EA1B0, EA1B1, s, t
    )
    results.append(formula_result)

    print(f"\n新公式计算结果 (s={s}, t={t}):")
    print(f"arcsin((E00 + {s}*B0)/(1 + {s}*A0)) = {term1:.8f} rad")
    print(f"arcsin((E01 + {s}*B1)/(1 + {s}*A0)) = {term2:.8f} rad")
    print(f"arcsin((E10 + {t}*B0)/(1 + {t}*A1)) = {term3:.8f} rad")
    print(f"arcsin((E11 + {t}*B1)/(1 + {t}*A1)) = {term4:.8f} rad")
    print(f"总和 (term1 + term2 + term3 - term4) = {formula_result:.8f}")
    print(f"与π的差值 = {abs(formula_result - np.pi):.8f}")

# 检查线性独立性
rank = check_linear_independence(results)
print(f"\n线性独立性分析:")
print(f"四种result值: {results}")
print(f"矩阵的秩为 {rank}")
if rank == 1:
    print("所有result值都是线性相关的（成比例）")
else:
    print(f"有 {rank} 个线性无关的result值")