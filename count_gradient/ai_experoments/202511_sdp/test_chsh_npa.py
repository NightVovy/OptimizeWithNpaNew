import numpy as np
import sys
import os
import unittest


# --- 1. FC2CG 函数 (直接包含) ---
def FC2CG(FC: np.ndarray, behaviour: int = 0) -> np.ndarray:
    """
    将 Full Correlator 矩阵转换为 Collins-Gisin 格式。
    """
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
    else:
        raise ValueError("参数 'behaviour' 必须是 0 或 1。")

    return CG


# --- 2. 测试类 (适配 unittest 和 pytest) ---
class TestNPAHierarchy(unittest.TestCase):

    def test_chsh_npa(self):
        """
        测试 CHSH 相关性是否满足 NPA 层次结构 (Level 3)。
        函数名以 test_ 开头，这样 pytest 才能识别。
        """
        # 尝试导入 NPAHierarchy
        try:
            from NPAHierarchy import npa_hierarchy
        except ImportError as e:
            self.fail(f"无法导入 npa_hierarchy，请确保文件在当前目录。错误: {e}")

        print("\n=== 开始 NPA Hierarchy 测试 (CHSH Tsirelson Bound) ===\n")

        # 1. 定义相关性参数 (Tsirelson Bound)
        a0 = 0
        a1 = 0
        b0 = 0
        b1 = 0
        e00 = 1 / np.sqrt(2)
        e01 = 1 / np.sqrt(2)
        e10 = 1 / np.sqrt(2)
        e11 = -1 / np.sqrt(2)

        # 2. 构造 Full Correlator 矩阵 M_fc
        M_fc = np.array([
            [0.0, b0, b1],
            [a0, e00, e01],
            [a1, e10, e11]
        ])

        # 3. 转换为 Collins-Gisin 格式
        M2cg = FC2CG(M_fc, behaviour=1)

        # 4. 定义描述向量 [OA, OB, MA, MB]
        desc = np.array([2, 2, 2, 2])
        print(f"Desc: {desc}")

        # 5. 运行 NPA Hierarchy 检查 (Level 3)
        k_level = 3
        print(f"Checking Level K={k_level}...")

        is_npa = npa_hierarchy(M2cg, desc, k=k_level)

        print(f"Result IS_NPA = {is_npa}")

        # 断言结果：应该是 1.0 (量子相关)
        self.assertAlmostEqual(is_npa, 1.0, places=5, msg="CHSH Tsirelson bound 应该通过 NPA 测试")

        if is_npa > 0.99:
            print("✅ 成功: Compatible with Quantum Theory.")
        else:
            print("❌ 失败: Incompatible.")


if __name__ == '__main__':
    unittest.main()