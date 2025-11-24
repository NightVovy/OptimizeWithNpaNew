import numpy as np


def FC2CG(FC: np.ndarray, behaviour: int = 0) -> np.ndarray:
    """
    将 Bell 泛函或行为的 Full Correlator 矩阵转换为 Collins-Gisin 格式。

    Full Correlator (FC) 格式:
     FC = [  K     <B1>   <B2>   ...
           <A1>  <A1B1> <A1B2> ...
           ...     ...    ...   ... ]

    Collins-Gisin (CG) 格式:
     CG = [  K     pB(0|1) pB(0|2) ...
           pA(0|1) p(00|11) p(00|12) ...
           ...       ...      ...    ... ]

    参数:
        FC (np.ndarray): 输入的 Full Correlator 矩阵。
        behaviour (int):
            - 0 (默认): 将 FC 视为 **Bell 泛函** (Bell functional)。
            - 1: 将 FC 视为 **行为** (behaviour/joint probabilities)。

    返回:
        np.ndarray: 转换后的 Collins-Gisin 矩阵。
    """

    # 检查输入是否为二维矩阵
    if FC.ndim != 2:
        raise ValueError("输入矩阵 FC 必须是二维的。")

    # 获取 FC 的尺寸
    # ia 是 A 方的测量数 (行数 - 1)
    # ib 是 B 方的测量数 (列数 - 1)
    ia, ib = FC.shape[0] - 1, FC.shape[1] - 1

    # 初始化 CG 矩阵
    CG = np.zeros((ia + 1, ib + 1))

    # 提取 FC 的子矩阵
    # A 是 <Ai> (FC 的第 2 行到最后一行, 第 1 列)
    A = FC[1:, 0].reshape(-1, 1)  # 转化为列向量 (ia x 1)
    # B 是 <Bj> (FC 的第 1 行, 第 2 列到最后一列)
    B = FC[0, 1:].reshape(1, -1)  # 转化为行向量 (1 x ib)
    # C 是 <AiBj> (FC 的关联项矩阵)
    C = FC[1:, 1:]  # (ia x ib)
    # K 是常数项 (FC[0, 0])

    if behaviour == 0:
        # --- 转换为 Bell Functional (Bell 泛函) 格式 ---

        # K -> CG(1,1)
        # K' = K + sum(<AiBj>) - sum(<Ai>) - sum(<Bj>)
        CG[0, 0] = FC[0, 0] + np.sum(C) - np.sum(A) - np.sum(B)

        # <Ai> -> CG(i+1, 1)
        # pA(0|i) = (1 + <Ai>)/2, 但这里是泛函，使用 2*<Ai> - 2*sum_j(<AiBj>)
        CG[1:, 0] = (2 * A - 2 * np.sum(C, axis=1, keepdims=True)).flatten()

        # <Bj> -> CG(1, j+1)
        # pB(0|j) = (1 + <Bj>)/2, 但这里是泛函，使用 2*<Bj> - 2*sum_i(<AiBj>)
        CG[0, 1:] = (2 * B - 2 * np.sum(C, axis=0, keepdims=True)).flatten()

        # <AiBj> -> CG(i+1, j+1)
        # p(00|ij) = (1 + <Ai> + <Bj> + <AiBj>)/4, 但这里是泛函，使用 4*<AiBj>
        CG[1:, 1:] = 4 * C

    elif behaviour == 1:
        # --- 转换为 Behaviour (行为/联合概率) 格式 ---

        # K -> CG(1,1)
        # 总是 1，因为联合概率矩阵的和为 1
        CG[0, 0] = 1.0

        # <Ai> -> CG(i+1, 1)
        # pA(0|i) = (1 + <Ai>)/2
        CG[1:, 0] = ((1 + A) / 2).flatten()

        # <Bj> -> CG(1, j+1)
        # pB(0|j) = (1 + <Bj>)/2
        CG[0, 1:] = ((1 + B) / 2).flatten()

        # <AiBj> -> CG(i+1, j+1)
        # p(00|ij) = (1 + <Ai> + <Bj> + <AiBj>)/4
        # A 是 (ia x 1)， B 是 (1 x ib)
        # numpy 会自动进行广播 (Broadcasting)
        # ones_ia_ib = np.ones((ia, ib)) # 可以不显式创建

        CG[1:, 1:] = (1 + A + B + C) / 4

    else:
        raise ValueError("参数 'behaviour' 必须是 0 (Bell 泛函) 或 1 (行为)。")

    return CG

# 示例调用 (在实际使用中，不需要这部分，但有助于理解和验证)
if __name__ == '__main__':
    # 示例 FC 矩阵 (2个A测量, 2个B测量, e.g., CHSH, FC是 3x3)
    # 注意输入的是correlations 即点P的8位
    a0 = 0
    a1 = 0
    b0 = 0
    b1 = 0
    e00 = 1 / np.sqrt(2)
    e01 = 1 / np.sqrt(2)
    e10 = 1 / np.sqrt(2)
    e11 = -1 / np.sqrt(2)

    fc_example = np.array([
        [0.0, b0, b1],
        [a0, e00, e01],
        [a1, e10, e11]
    ])

#     # 1. Bell 泛函转换 (behaviour=0)
    cg_functional = FC2CG(fc_example, behaviour=0)
    print("--- Bell Functional (CG) ---")
    print(cg_functional)
#     # 预期结果 (近似):
#     # [[ 0.4, -0.4,  0.4],
#     #  [ 0.6,  1.6, -1.6],
#     #  [-0.6, -1.6,  1.6]]

#     # 2. Behaviour 转换 (behaviour=1)
    cg_behaviour = FC2CG(fc_example, behaviour=1)
    print("\n--- Behaviour (CG) ---")
    print(cg_behaviour)
#     # 预期结果 (近似):
#     # [[1.0, 0.8, 0.2],
#     #  [0.75, 0.625, 0.025],
#     #  [0.25, 0.025, 0.625]]