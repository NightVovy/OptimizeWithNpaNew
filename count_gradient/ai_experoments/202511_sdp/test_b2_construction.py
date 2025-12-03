import numpy as np
import cvxpy as cp
import sys
import unittest

# 导入依赖
try:
    from NPAHierarchy_3 import npa_constraints
    from locate_gamma_2 import get_gamma_element
    from npa_support import find_in_cell
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


# ... (create_extended_coeff_matrix 和 build_adaptive_bell_expression 函数保持不变) ...
def create_extended_coeff_matrix():
    M = np.zeros((3, 4))
    M[1, 0] = 0.5;
    M[1, 1] = 1.0;
    M[1, 2] = 1.0
    M[2, 1] = 1.0;
    M[2, 2] = -1.0
    return M


def build_adaptive_bell_expression(G, ind_catalog, m_vec, coeff_matrix):
    # ... (保持原代码不变) ...
    # 为了节省篇幅，这里省略函数体，请保留您原来的代码
    expr = 0
    num_alice = coeff_matrix.shape[0] - 1
    num_bob = coeff_matrix.shape[1] - 1

    def get_term_E(alice_list, bob_list):
        t = get_gamma_element(G, ind_catalog, m_vec, alice_list, bob_list)
        if t is None: raise ValueError(f"Gamma 矩阵中缺少项: A{alice_list}B{bob_list}")
        return t

    if coeff_matrix[0, 0] != 0: expr += coeff_matrix[0, 0]
    for y_idx in range(num_bob):
        c = coeff_matrix[0, y_idx + 1]
        if c != 0: expr += c * (2 * get_term_E([], [m_vec[0] + y_idx]) - 1)
    for x_idx in range(num_alice):
        c = coeff_matrix[x_idx + 1, 0]
        if c != 0: expr += c * (2 * get_term_E([x_idx], []) - 1)
    for x_idx in range(num_alice):
        for y_idx in range(num_bob):
            c = coeff_matrix[x_idx + 1, y_idx + 1]
            if c != 0:
                bob_idx = m_vec[0] + y_idx
                t_AB = 4 * get_term_E([x_idx], [bob_idx]) - 2 * get_term_E([x_idx], []) - 2 * get_term_E([],
                                                                                                         [bob_idx]) + 1
                expr += c * t_AB
    return expr


class TestB2Construction(unittest.TestCase):

    def test_run_b2_check(self):
        desc = np.array([2, 2, 2, 3])
        m_vec = desc[2:4]  # [2, 3]
        k_level = 3

        print(f"Scenario: {desc}")

        # 1. 构建变量
        dummy_cg = np.zeros((3, 4))
        G, constraints, ind_catalog = npa_constraints(dummy_cg, desc, k=k_level, enforce_data=False)

        print(f"Gamma Matrix Size: {G.shape}")  # 此时 G 只是符号，没有值

        # 2. 验证 B2 存在 (逻辑验证)
        idx_b2 = find_in_cell(np.array([[4], [0]]), ind_catalog)
        self.assertNotEqual(idx_b2, -1, "Bob's setting B2 (Index 4) not found!")
        print("Structure check passed.")

        # ==========================================
        # [新增] 3. 执行求解，赋予 G 数值
        # ==========================================
        print("\nSolving dummy problem to populate G values...")
        # 我们随便最小化一个东西，或者最小化 0，只为了让求解器跑起来
        # 注意：因为 enforce_data=False，且没有施加 Bell 约束，这是一个松弛问题，肯定有解
        prob = cp.Problem(cp.Minimize(0), constraints)
        try:
            prob.solve(solver=cp.SCS, eps=1e-4)  # 精度不用太高，只要能算出数就行
        except cp.SolverError:
            print("Solver Error (Expected if constraints are bad, but here should be fine)")

        print(f"Solve Status: {prob.status}")

        # ==========================================
        # [DEBUG POINT] 在这里打断点
        # ==========================================
        print("\n现在 G.value 应该有值了。请在此处打断点查看。")

        # 此时你可以右键 G.value -> View as Array
        # 并且你可以验证 ind_catalog 确实对应了 G 的行列

        dummy_stop = 0


if __name__ == "__main__":
    unittest.main()