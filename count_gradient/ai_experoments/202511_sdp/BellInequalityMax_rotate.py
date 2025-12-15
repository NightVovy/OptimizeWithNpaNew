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
    """

    # 1. 参数解析
    desc = np.array(desc, dtype=int).flatten()
    oa, ob, ma, mb = desc[0], desc[1], desc[2], desc[3]
    M = np.array(coefficients)

    # 2. 格式转换逻辑 (支持 fc -> cg)
    if notation.lower() == 'fc':
        if oa != 2 or ob != 2:
            raise ValueError("Full Correlator ('fc') 格式仅支持 2 个输出 (OA=OB=2) 的场景。")

        if M.shape != (ma + 1, mb + 1):
            raise ValueError(f"FC 矩阵维度错误。期望: ({ma + 1}, {mb + 1}), 实际: {M.shape}")

        try:
            M = FC2CG(M, behaviour=0)
            notation = 'cg'
        except Exception as e:
            raise RuntimeError(f"FC 转 CG 失败: {e}")

    elif notation.lower() == 'cg':
        expected_shape = ((oa - 1) * ma + 1, (ob - 1) * mb + 1)
        if M.shape != expected_shape:
            raise ValueError(f"CG 系数矩阵维度错误。期望: {expected_shape}, 实际: {M.shape}")

    else:
        raise NotImplementedError(f"不支持的表示法: '{notation}'。")

    # ==========================================================================
    # CASE 1: Quantum Maximum (SDP via NPAHierarchy)
    # ==========================================================================
    if mtype.lower() == 'quantum':
        p_cg = cp.Variable(M.shape)
        objective = cp.Maximize(cp.sum(cp.multiply(M, p_cg)))

        constraints = []
        constraints.append(p_cg[0, 0] == 1)

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
        alice_strategies = list(itertools.product(range(oa), repeat=ma))
        bob_strategies = list(itertools.product(range(ob), repeat=mb))

        max_val = -np.inf

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

        for a_strat in alice_strategies:
            for b_strat in bob_strategies:
                current_val = 0.0
                current_val += M[0, 0]  # K

                for y in range(mb):  # Bob marginals
                    b_choice = b_strat[y]
                    if b_choice < ob - 1:
                        _, col = get_cg_indices(None, None, b_choice, y)
                        if col is not None: current_val += M[0, col]

                for x in range(ma):  # Alice marginals
                    a_choice = a_strat[x]
                    if a_choice < oa - 1:
                        row, _ = get_cg_indices(a_choice, x, None, None)
                        if row is not None: current_val += M[row, 0]

                for x in range(ma):  # Correlations
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
# 测试代码 / 主程序入口
# ==============================================================================
if __name__ == '__main__':
    print("=== Tilted-CHSH Inequality Max Value Calculation ===")

    # 场景: 2输出, 2输入
    desc = [2, 2, 2, 2]

    # 1. 设定 theta
    # 可以修改此处的值，例如 np.pi/6, np.arcsin(1/np.sqrt(3))/2 等
    theta_rad = np.pi / 6

    # 2. 计算相关参数
    # sin(2theta)
    sin_2theta = np.sin(2 * theta_rad)
    cos_2theta = np.cos(2 * theta_rad)

    # 计算 Alpha
    # 公式: sin^2(2theta) = (1 - alpha^2/4) / (1 + alpha^2/4)
    # 反解: alpha = 2 * cos(2theta) / sqrt(1 + sin^2(2theta))
    alpha_numerator = 2 * np.abs(cos_2theta)
    alpha_denominator = np.sqrt(1 + sin_2theta ** 2)
    alpha = alpha_numerator / alpha_denominator

    # 计算 Mu
    # 公式: tan(mu) = sin(2theta)
    mu = np.arctan(sin_2theta)

    # 计算 Mu 的三角函数
    sin_mu = np.sin(mu)
    cos_mu = np.cos(mu)
    sin_2mu = np.sin(2 * mu)
    cos_2mu = np.cos(2 * mu)

    # 打印参数
    print(f"\n[Parameters]")
    print(f"Theta (rad) : {theta_rad:.6f}")
    print(f"Alpha       : {alpha:.6f}")
    print(f"Mu (rad)    : {mu:.6f}")
    print(f"sin(2theta) : {sin_2theta:.6f}")

    # 3. 计算系数项
    # A0 项系数
    coeff_A0 = alpha
    # A0B0 项系数
    coeff_A0B0 = 2 * cos_mu
    # A1B0 项系数
    coeff_A1B0 = cos_2mu / cos_mu
    # A1B1 项系数
    coeff_A1B1 = -1.0 / cos_mu

    print(f"\n[Coefficients]")
    print(f"A0   (alpha)           : {coeff_A0:.6f}")
    print(f"A0B0 (2*cos_mu)        : {coeff_A0B0:.6f}")
    print(f"A1B0 (cos2mu/cosmu)    : {coeff_A1B0:.6f}")
    print(f"A1B1 (-1/cosmu)        : {coeff_A1B1:.6f}")

    # 4. 构建 M_fc 矩阵
    # 结构:
    #      K   B0  B1
    #  K   0   0   0
    # A0   c1  c2  0
    # A1   0   c3  c4

    M_fc = np.zeros((3, 3))

    # 填入非零项
    M_fc[1, 0] = coeff_A0  # <A0>
    M_fc[1, 1] = coeff_A0B0  # <A0B0>
    M_fc[1, 2] = 0  # <A0B1> (按要求设为0)

    M_fc[2, 0] = 0  # <A1>
    M_fc[2, 1] = coeff_A1B0  # <A1B0>
    M_fc[2, 2] = coeff_A1B1  # <A1B1>

    print("\n[Input Matrix (FC Format)]")
    print(M_fc)

    # 5. 计算最大值
    try:
        # Classical Max
        b_cl = bell_inequality_max(M_fc, desc, notation='fc', mtype='classical')
        print(f"\nClassical Max Bound : {b_cl:.8f}")

        # Quantum Max (Theoretical check)
        # 理论上 Tilted CHSH 的量子最大值应该是 sqrt(8 + 2*alpha^2)
        # (如果是标准形式)。对于旋转后的形式，物理本质不变，最大值应相同。
        theo_q_bound = np.sqrt(8 + 2 * alpha ** 2)
        print(f"Theoretical Q Bound : {theo_q_bound:.8f} (sqrt(8+2*alpha^2))")

        # Quantum Max (SDP Calculation)
        # 使用 Level 1+ab 应该足够 (对应 2-2-2-2 场景)
        b_qm = bell_inequality_max(M_fc, desc, notation='fc', mtype='quantum', k='3')
        print(f"Calculated Q Bound  : {b_qm:.8f}")

    except Exception as e:
        print(f"计算出错: {e}")