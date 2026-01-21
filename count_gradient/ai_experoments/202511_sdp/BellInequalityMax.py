import numpy as np
import cvxpy as cp
import sys
import itertools

# ==============================================================================
# [导入依赖]
# ==============================================================================
try:
    # 导入 NPA 构建器
    from NPAHierarchy_5 import npa_constraints
    # 导入格式转换工具 FC2CG (用于支持 'fc' 格式)
    from fc2cg import FC2CG
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 NPAHierarchy_5.py 和 fc2cg.py 都在当前目录下。")
    sys.exit(1)


def bell_inequality_max(coefficients, desc, notation='cg', mtype='classical', k=1, solver=None, verbose=False):
    """
    计算 Bell 不等式的最大值 (Classical / Quantum)。
    (注: No-Signaling 计算已被移除)

    参数:
        coefficients: numpy.ndarray
            Bell 不等式的系数矩阵。
            - 如果 notation='cg': ((OA-1)*MA+1, (OB-1)*MB+1)
            - 如果 notation='fc': (MA+1, MB+1)
        desc: list or numpy.ndarray
            场景描述向量 [OA, OB, MA, MB]。
            OA/OB: Alice/Bob 输出数量 (必须为 2 才能使用 FC 格式)。
            MA/MB: Alice/Bob 输入数量。
        notation: str, default 'cg'
            系数的表示法。支持 'cg' (Collins-Gisin) 和 'fc' (Full Correlator)。
        mtype: str, default 'classical'
            计算类型: 'classical' 或 'quantum'。
        k: int or str, default 1
            NPA 层级。仅在 mtype='quantum' 时有效。
        solver: str, optional
            CVXPY 求解器 (例如 'MOSEK', 'SCS')。
        verbose: bool
            是否打印求解过程。

    返回:
        bmax: float
            计算得到的 Bell 不等式最大值。
    """

    # 1. 参数解析
    desc = np.array(desc, dtype=int).flatten()
    oa, ob, ma, mb = desc[0], desc[1], desc[2], desc[3]
    M = np.array(coefficients)

    # 2. 格式转换逻辑 (支持 fc -> cg)
    if notation.lower() == 'fc':
        # FC 格式只适用于二值输出 (OA=OB=2)
        if oa != 2 or ob != 2:
            raise ValueError("Full Correlator ('fc') 格式仅支持 2 个输出 (OA=OB=2) 的场景。")

        # 检查 FC 矩阵维度应为 (MA+1, MB+1)
        if M.shape != (ma + 1, mb + 1):
            raise ValueError(f"FC 矩阵维度错误。期望: ({ma + 1}, {mb + 1}), 实际: {M.shape}")

        # 调用转换函数: behaviour=0 表示这是一个 Bell 泛函
        try:
            M = FC2CG(M, behaviour=0)
            # 转换后，notation 视为 'cg' 继续处理
            notation = 'cg'
        except Exception as e:
            raise RuntimeError(f"FC 转 CG 失败: {e}")

    elif notation.lower() == 'cg':
        # 验证 CG 维度
        expected_shape = ((oa - 1) * ma + 1, (ob - 1) * mb + 1)
        if M.shape != expected_shape:
            raise ValueError(f"CG 系数矩阵维度错误。期望: {expected_shape}, 实际: {M.shape}")

    else:
        raise NotImplementedError(
            f"不支持的表示法: '{notation}'。目前仅支持 'cg' 和 'fc'。"
        )

    # ==========================================================================
    # CASE 1: Quantum Maximum (SDP via NPAHierarchy)
    # ==========================================================================
    if mtype.lower() == 'quantum':
        # 定义变量 P (CG 格式)
        p_cg = cp.Variable(M.shape)

        # 目标函数
        objective = cp.Maximize(cp.sum(cp.multiply(M, p_cg)))

        constraints = []
        # 1. 归一化约束
        constraints.append(p_cg[0, 0] == 1)

        # 2. NPA 层级约束
        try:
            _, npa_cons, _ = npa_constraints(p_cg, desc, k=k, enforce_data=True)
            constraints.extend(npa_cons)
        except Exception as e:
            raise RuntimeError(f"构建 NPA 约束时出错: {e}")

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=solver, verbose=verbose)
            status = prob.status
            if status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return prob.value
            else:
                print(f"警告: 求解状态为 {status}")
                return prob.value if prob.value is not None else -np.inf
        except cp.SolverError as e:
            print(f"求解器错误: {e}")
            return None

    # ==========================================================================
    # CASE 2: Classical Maximum (Brute Force)
    # ==========================================================================
    elif mtype.lower() == 'classical':
        # 经典最大值：遍历所有确定性策略

        # 生成策略组合
        alice_strategies = list(itertools.product(range(oa), repeat=ma))
        bob_strategies = list(itertools.product(range(ob), repeat=mb))

        max_val = -np.inf

        # 辅助函数: 获取 CG 索引
        def get_cg_indices(a_out, a_in, b_out, b_in):
            row = 0
            col = 0
            if a_in is not None:
                if a_out < oa - 1:
                    row = 1 + a_out + a_in * (oa - 1)
                else:
                    return None, None
            if b_in is not None:
                if b_out < ob - 1:
                    col = 1 + b_out + b_in * (ob - 1)
                else:
                    return None, None
            return row, col

        # 暴力遍历
        for a_strat in alice_strategies:
            for b_strat in bob_strategies:
                current_val = 0.0

                # M[0,0] * 1
                current_val += M[0, 0]

                # Bob 边缘项
                for y in range(mb):
                    b_choice = b_strat[y]
                    if b_choice < ob - 1:
                        _, col = get_cg_indices(None, None, b_choice, y)
                        if col is not None: current_val += M[0, col]

                # Alice 边缘项
                for x in range(ma):
                    a_choice = a_strat[x]
                    if a_choice < oa - 1:
                        row, _ = get_cg_indices(a_choice, x, None, None)
                        if row is not None: current_val += M[row, 0]

                # 联合项
                for x in range(ma):
                    for y in range(mb):
                        a_choice = a_strat[x]
                        b_choice = b_strat[y]
                        if a_choice < oa - 1 and b_choice < ob - 1:
                            row, col = get_cg_indices(a_choice, x, b_choice, y)
                            if row is not None and col is not None:
                                current_val += M[row, col]

                if current_val > max_val:
                    max_val = current_val

        return max_val

    else:
        raise ValueError("mtype 参数必须是 'classical' 或 'quantum'。")


# ==============================================================================
# 测试代码
# ==============================================================================
if __name__ == '__main__':
    print("=== 测试 BellInequalityMax ('fc' 支持, 无 nosignal) ===")

    # 场景: CHSH (2,2,2,2)
    desc_chsh = [2, 2, 2, 2]

    # 构造标准 CHSH 系数 (FC 格式)
    # CHSH Operator = <A0B0> + <A0B1> + <A1B0> - <A1B1>
    # M_fc = np.array([
    #     [0, 0, 0],
    #     [0.75592895, 1, 1],
    #     [0, 1, -1]
    # ], dtype=float)
    M_fc = np.array([
        [0, 0, 0.9084],
        [0, 0, 0],
        [0.3368, 0, 1]
    ], dtype=float)

    print("输入 FC 矩阵 :")
    print(M_fc)

    try:
        # 1. Classical Max (理论值 2.0)
        b_cl = bell_inequality_max(M_fc, desc_chsh, notation='fc', mtype='classical')
        print(f"\n[Classical Max]: {b_cl}")

        # 2. Quantum Max (理论值 2*sqrt(2) ≈ 2.8284)
        b_qm = bell_inequality_max(M_fc, desc_chsh, notation='fc', mtype='quantum', k=3)
        print(f"[Quantum Max (L1)]: {b_qm}")

    except Exception as e:
        print(f"测试出错: {e}")