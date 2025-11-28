import numpy as np
import unittest


# 假设你的函数保存在 npa_utils.py 中，如果不是，请修改 import
from product_of_orthogonal import product_of_orthogonal

# 为了方便直接运行，这里直接引用你提供的函数定义
# (在实际使用中，请确保 product_of_orthogonal 函数已定义在当前作用域或被 import)

class TestProductOfOrthogonal(unittest.TestCase):
    def setUp(self):
        # 定义实验场景: Alice 2个设置, Bob 2个设置
        # Alice 索引: 0, 1
        # Bob 索引: 2, 3
        self.m_vec = np.array([2, 2])

    def test_single_operator(self):
        """测试 1: 单一算符 (Type 1)"""
        print("\n--- 测试 1: 单一算符 ---")
        # Alice 测量 0, 结果 0
        ind = np.array([[0], [0]])
        res, res_type = product_of_orthogonal(ind, self.m_vec)

        print(f"Input:\n{ind}\nOutput:\n{res}\nType: {res_type}")
        self.assertEqual(res_type, 1)
        np.testing.assert_array_equal(res, ind)

    def test_identity_simplification(self):
        """测试 2: 恒等算符简化 (I * A -> A)"""
        print("\n--- 测试 2: 恒等算符简化 ---")
        # [I, A0_0] -> [0, 0; -1, 0]
        ind = np.array([[0, 0], [-1, 0]])
        res, res_type = product_of_orthogonal(ind, self.m_vec)

        expected_res = np.array([[0], [0]])
        print(f"Input:\n{ind}\nOutput:\n{res}\nType: {res_type}")
        self.assertEqual(res_type, 1)
        np.testing.assert_array_equal(res, expected_res)

        # 测试全部消除 [I, I] -> I
        ind_all_I = np.array([[0, 0], [-1, -1]])
        res_I, res_type_I = product_of_orthogonal(ind_all_I, self.m_vec)
        print(f"Input (All I):\n{ind_all_I}\nOutput:\n{res_I}\nType: {res_type_I}")
        self.assertEqual(res_type_I, 1)
        # 结果应该是标准的恒等算符 [0; -1]
        np.testing.assert_array_equal(res_I, np.array([[0], [-1]]))

    def test_orthogonality(self):
        """测试 3: 正交性 (A0_0 * A0_1 -> 0) (Type 0)"""
        print("\n--- 测试 3: 正交性 ---")
        # 同一设置 (0)，不同结果 (0 和 1)
        ind = np.array([[0, 0], [0, 1]])
        res, res_type = product_of_orthogonal(ind, self.m_vec)

        print(f"Input:\n{ind}\nOutput:\n{res}\nType: {res_type}")
        self.assertEqual(res_type, 0)
        # 零算符通常返回 [[0], [-1]] 占位
        np.testing.assert_array_equal(res, np.array([[0], [-1]]))

    def test_idempotency(self):
        """测试 4: 幂等性 (A0_0 * A0_0 -> A0_0)"""
        print("\n--- 测试 4: 幂等性 ---")
        ind = np.array([[0, 0], [0, 0]])
        res, res_type = product_of_orthogonal(ind, self.m_vec)

        expected_res = np.array([[0], [0]])
        print(f"Input:\n{ind}\nOutput:\n{res}\nType: {res_type}")
        self.assertEqual(res_type, 1)
        np.testing.assert_array_equal(res, expected_res)

    def test_commutativity_alice_bob(self):
        """测试 5: 交换律 (B * A -> A * B) (Type 2)"""
        print("\n--- 测试 5: 交换律 (Bob * Alice) ---")
        # Bob (设置2, 结果0) * Alice (设置0, 结果0)
        # ind = [[2, 0], [0, 0]]
        ind = np.array([[2, 0], [0, 0]])
        res, res_type = product_of_orthogonal(ind, self.m_vec)

        # 期望: Alice 在前 [[0, 2], [0, 0]]
        expected_res = np.array([[0, 2], [0, 0]])
        print(f"Input:\n{ind}\nOutput:\n{res}\nType: {res_type}")
        self.assertEqual(res_type, 2)
        np.testing.assert_array_equal(res, expected_res)

    def test_non_commuting_barrier(self):
        """测试 6: 同方阻隔 (A0 * A1 * B0 -> 不可交换 A0 A1)"""
        print("\n--- 测试 6: 同方阻隔 (非交换) ---")
        # Alice0 * Alice1 * Bob0
        # A0 和 A1 是 Alice 的不同设置，非对易，不可简化
        ind = np.array([[0, 1, 2], [0, 0, 0]])
        res, res_type = product_of_orthogonal(ind, self.m_vec)

        print(f"Input:\n{ind}\nOutput:\n{res}\nType: {res_type}")
        # 应该保持原样，无法简化为 Type 1 或 2，返回 Type -1
        self.assertEqual(res_type, -1)
        np.testing.assert_array_equal(res, ind)

    def test_complex_simplification(self):
        """测试 7: 复杂链式简化 (B0 * A0 * A0 * B0 -> A0 * B0)"""
        print("\n--- 测试 7: 复杂链式简化 ---")
        # 序列: Bob0 * Alice0 * Alice0 * Bob0
        # 1. 中间 A0*A0 -> A0 (幂等) ==> B0 * A0 * B0
        # 2. B0 * A0 -> A0 * B0 (交换) ==> A0 * B0 * B0
        # 3. B0 * B0 -> B0 (幂等) ==> A0 * B0
        ind = np.array([[2, 0, 0, 2], [0, 0, 0, 0]])

        res, res_type = product_of_orthogonal(ind, self.m_vec)

        # 期望最终结果: A0 * B0 (Type 2)
        expected_res = np.array([[0, 2], [0, 0]])

        print(f"Input:\n{ind}\nOutput:\n{res}\nType: {res_type}")
        self.assertEqual(res_type, 2)
        np.testing.assert_array_equal(res, expected_res)

    def test_zero_propagation(self):
        """测试 8: 零传播 (A0_0 * B0 * A0_1 -> 0)"""
        print("\n--- 测试 8: 零传播 ---")
        # 序列: A0(结果0) * B0 * A0(结果1)
        # B0 和后面的 A0(1) 交换 -> A0(0) * A0(1) * B0
        # A0(0) * A0(1) 正交 -> 0
        ind = np.array([[0, 2, 0], [0, 0, 1]])

        res, res_type = product_of_orthogonal(ind, self.m_vec)

        print(f"Input:\n{ind}\nOutput:\n{res}\nType: {res_type}")
        self.assertEqual(res_type, 0)
        # 零算符
        np.testing.assert_array_equal(res, np.array([[0], [-1]]))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)