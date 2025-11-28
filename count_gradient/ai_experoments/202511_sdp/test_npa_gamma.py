import numpy as np
import sys
import os
import unittest

# 导入您修改后的主函数
# 假设文件名是 NPAHierarchy_with_gamma.py
try:
    from NPAHierarchy_with_gamma import npa_hierarchy
except ImportError:
    print("错误: 无法找到 NPAHierarchy_with_gamma.py，请确保文件在当前目录下。")
    # 注意：在 unittest 中不能直接 sys.exit，会中断测试框架
    # 这里我们留给后面的 assert 处理


# --- 辅助函数: FC2CG ---
def FC2CG(FC: np.ndarray, behaviour: int = 0) -> np.ndarray:
    if FC.ndim != 2:
        raise ValueError("输入矩阵 FC 必须是二维的。")
    ia, ib = FC.shape[0] - 1, FC.shape[1] - 1
    CG = np.zeros((ia + 1, ib + 1))
    A = FC[1:, 0].reshape(-1, 1)
    B = FC[0, 1:].reshape(1, -1)
    C = FC[1:, 1:]
    if behaviour == 0:
        CG[0, 0] = FC[0, 0] + np.sum(C) - np.sum(A) - np.sum(B)
        CG[1:, 0] = (2 * A - 2 * np.sum(C, axis=1, keepdims=True)).flatten()
        CG[0, 1:] = (2 * B - 2 * np.sum(C, axis=0, keepdims=True)).flatten()
        CG[1:, 1:] = 4 * C
    elif behaviour == 1:
        CG[0, 0] = 1.0
        CG[1:, 0] = ((1 + A) / 2).flatten()
        CG[0, 1:] = ((1 + B) / 2).flatten()
        CG[1:, 1:] = (1 + A + B + C) / 4
    return CG


# --- 测试类 ---
class TestNPAGamma(unittest.TestCase):

    def test_run_npa_gamma(self):
        print("\n=== NPA Hierarchy Test (Generating Gamma Matrix) ===")

        # 1. 定义输入参数
        a0 = 0
        a1 = 0
        b0 = 0
        b1 = 0
        e00 = 1 / np.sqrt(2)
        e01 = 1 / np.sqrt(2)
        e10 = 1 / np.sqrt(2)
        e11 = -1 / np.sqrt(2)

        # 2. 构造 Full Correlator 矩阵
        M_fc = np.array([
            [0.0, b0, b1],
            [a0, e00, e01],
            [a1, e10, e11]
        ])

        # 3. 转换为 Collins-Gisin 格式
        M2cg = FC2CG(M_fc, behaviour=1)

        # 4. 定义描述向量
        desc = np.array([2, 2, 2, 2])

        # 5. 调用 NPA Hierarchy
        print("Running NPA Hierarchy (K=3)...")
        is_npa, gamma_matrix, ind_catalog = npa_hierarchy(M2cg, desc, k=3)

        # 6. 输出与调试锚点
        print(f"\n=== 结果 ===")
        print(f"Satisfies NPA? {is_npa}")

        # 简单断言确保运行成功
        self.assertIsNotNone(gamma_matrix, "Gamma 矩阵生成失败")
        self.assertGreater(len(ind_catalog), 0, "Index Catalog 为空")

        print(f"Gamma Matrix Shape: {gamma_matrix.shape}")
        print(f"Index Catalog Length: {len(ind_catalog)}")

        # ==========================================
        # [DEBUG POINT] 在下面这一行打红点断点
        # ==========================================
        print("请在此处打断点，然后在 Debugger 中查看 gamma_matrix 和 ind_catalog")

        # 仅仅为了让代码有点事情做，方便断点停留
        dummy_var = 1


if __name__ == "__main__":
    unittest.main()