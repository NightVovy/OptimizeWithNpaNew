import unittest
import numpy as np

# 导入你的辅助函数模块
# 确保 npa_support.py 在同一目录下，或者在 Python 路径中
from npa_support import find_in_cell, update_ind


class TestNPASupport(unittest.TestCase):

    def test_find_in_cell(self):
        """
        测试 find_in_cell 函数的功能:
        1. 能找到存在的数组并返回正确索引 (基于 0)。
        2. 找不到时返回 -1。
        """
        print("\n--- 测试 find_in_cell ---")

        # 准备数据目录
        arr1 = np.array([[0], [0]])
        arr2 = np.array([[0, 1], [0, 0]])
        arr3 = np.array([[1, 2], [1, 1]])

        catalog = [arr1, arr2, arr3]

        # 测试用例 1: 查找第一个元素
        idx1 = find_in_cell(np.array([[0], [0]]), catalog)
        print(f"查找 arr1: 期望 0, 实际 {idx1}")
        self.assertEqual(idx1, 0)

        # 测试用例 2: 查找中间元素
        idx2 = find_in_cell(np.array([[0, 1], [0, 0]]), catalog)
        print(f"查找 arr2: 期望 1, 实际 {idx2}")
        self.assertEqual(idx2, 1)

        # 测试用例 3: 查找不存在的元素
        idx_none = find_in_cell(np.array([[9, 9], [9, 9]]), catalog)
        print(f"查找不存在元素: 期望 -1, 实际 {idx_none}")
        self.assertEqual(idx_none, -1)

        # 测试用例 4: 查找形状不同但存在的元素 (确保 array_equal 工作正常)
        idx3 = find_in_cell(np.array([[1, 2], [1, 1]]), catalog)
        self.assertEqual(idx3, 2)

    def test_update_ind(self):
        """
        测试 update_ind 函数的里程表逻辑。
        场景设定:
        Alice (MA=2, OA=2) -> m_vec[0]=2, o_vec[0]=1 (即 2-1)
        Bob   (MB=2, OB=2) -> m_vec[1]=2, o_vec[1]=1 (即 2-1)
        K = 2 (乘积长度)
        """
        print("\n--- 测试 update_ind ---")

        # 设置参数
        k = 2
        m_vec = np.array([2, 2])  # Alice 2个设置, Bob 2个设置
        o_vec = np.array([1, 1])  # 结果上限索引 (Outcomes - 1)

        # --- Case 1: 恒等算符初始化 ---
        # 输入: 第一行任意(通常0)，第二行全 -1
        # 预期: 全 0 (第一个非恒等测量)
        print("Case 1: 恒等算符初始化")
        old_ind = np.array([[0, 0], [-1, -1]])
        expected = np.array([[0, 0], [0, 0]])
        result = update_ind(old_ind, k, m_vec, o_vec)
        print(f"Input:\n{old_ind}\nOutput:\n{result}")
        np.testing.assert_array_equal(result, expected)

        # --- Case 2: 简单结果递增 ---
        # 当前: [0, 0; 0, 0] (Setting 0, Result 0)
        # 预期: [0, 0; 0, 1] (Setting 0, Result 1)
        print("\nCase 2: 简单结果递增")
        old_ind = np.array([[0, 0], [0, 0]])
        expected = np.array([[0, 0], [0, 1]])
        result = update_ind(old_ind, k, m_vec, o_vec)
        print(f"Input:\n{old_ind}\nOutput:\n{result}")
        np.testing.assert_array_equal(result, expected)

        # --- Case 3: 结果进位到设置 (Outcome Carry) ---
        # 当前: [0, 0; 0, 1] (Result 1 是上限)
        # 预期: [0, 1; 0, 0] (Setting 变成 1, Result 重置为 0)
        print("\nCase 3: 结果进位到设置")
        old_ind = np.array([[0, 0], [0, 1]])
        expected = np.array([[0, 1], [0, 0]])
        result = update_ind(old_ind, k, m_vec, o_vec)
        print(f"Input:\n{old_ind}\nOutput:\n{result}")
        np.testing.assert_array_equal(result, expected)

        # --- Case 4: 跨 Party 进位 (Alice -> Bob) ---
        # Alice 设置是 0, 1。Bob 是 2, 3。
        # 当前: [0, 1; 0, 1] (Setting 1 是 Alice 最后一个设置, Result 1 是上限)
        # 预期: [0, 2; 0, 0] (Setting 变成 2 [Bob], Result 重置为 0)
        print("\nCase 4: 跨 Party 进位")
        old_ind = np.array([[0, 1], [0, 1]])
        expected = np.array([[0, 2], [0, 0]])
        result = update_ind(old_ind, k, m_vec, o_vec)
        print(f"Input:\n{old_ind}\nOutput:\n{result}")
        np.testing.assert_array_equal(result, expected)

        # --- Case 5: 算符进位 (Operator Carry) ---
        # 总设置数 4 (0,1,2,3)。
        # 当前: [0, 3; 0, 1] (最后一个算符已经达到最大: Setting 3, Result 1)
        # 预期: [1, 0; 0, 0] (前一个算符的结果+1 -> [0,0]变为[0,1] X -> 其实是前一个算符 Result 0 变 1??
        # 等等，逻辑是: new_ind[1, l-1] += 1。
        # 如果前一个算符是 [0;0]，进位后变成 [0;1]。后一个算符重置为 [0;0]。
        # 修正预期: [0, 0; 0, 0] -> 进位 -> [0, 0; 1, 0]
        # 让我们手动推导一下:
        # l=1 (右边): new_ind[1,1]=1 -> 2 (Overflow outcome) -> new_ind[1,1]=0, new_ind[0,1]=3->4.
        # l=1: new_ind[0,1]=4 (Overflow settings sum(m_vec)=4).
        # l=1: new_ind[0,1]=0. 进位到左边 -> new_ind[1, 0] += 1.
        # 左边原为 [0;0], 变为 [0;1].
        # 结果: [[0, 0], [1, 0]]
        print("\nCase 5: 算符进位 (倒数第一位溢出，进位到倒数第二位)")
        old_ind = np.array([[0, 3], [0, 1]])
        expected = np.array([[0, 0], [1, 0]])
        result = update_ind(old_ind, k, m_vec, o_vec)
        print(f"Input:\n{old_ind}\nOutput:\n{result}")
        np.testing.assert_array_equal(result, expected)

        # --- Case 6: 边界溢出 (结束) ---
        # 假设所有位都满了，虽然 update_ind 不会自己判断“结束”，但它会返回溢出状态或者重置状态
        # 这是一个极端测试，确保不会报错
        print("\nCase 6: 极端溢出测试")
        # 两个算符都满了 [3, 3; 1, 1]
        old_ind = np.array([[3, 3], [1, 1]])
        # 推导:
        # l=1: [3;1] -> [3;2]溢出 -> [4;0]设置溢出 -> [0;0], 进位左边.
        # 左边 [3;1] 变为 [3;2] 溢出 -> [4;0] 设置溢出 -> [0;0], 进位到 l=-1 (无处可进，return)
        # 结果应该是 [[0, 0], [0, 0]] (类似于里程表归零)
        expected = np.array([[0, 0], [0, 0]])
        result = update_ind(old_ind, k, m_vec, o_vec)
        print(f"Input:\n{old_ind}\nOutput:\n{result}")
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()