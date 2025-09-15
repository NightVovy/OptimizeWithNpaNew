import numpy as np
from scipy.optimize import minimize, differential_evolution

# 给定的法向量 n3
n3 = np.array(['1.17588953', '1.17588951', '0.00000012', '0.00000019',
               '1.76383428', '-2.35177882', '1.76383427', '2.35177900'], dtype=float)

# 最大量子违背值
max_q_violation = 7.055255414815932


def compute_P4(params):
    """计算P4点坐标"""
    theta, a0, a1, b0, b1 = params
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)

    A0 = cos2t * np.cos(a0)
    A1 = cos2t * np.cos(a1)
    B0 = cos2t * np.cos(b0)
    B1 = cos2t * np.cos(b1)
    E00 = np.cos(a0) * np.cos(b0) + sin2t * np.sin(a0) * np.sin(b0)
    E01 = np.cos(a0) * np.cos(b1) + sin2t * np.sin(a0) * np.sin(b1)
    E10 = np.cos(a1) * np.cos(b0) + sin2t * np.sin(a1) * np.sin(b0)
    E11 = np.cos(a1) * np.cos(b1) + sin2t * np.sin(a1) * np.sin(b1)

    return np.array([A0, A1, B0, B1, E00, E01, E10, E11])


def objective(params):
    """目标函数：负的点积（用于最小化）"""
    P4 = compute_P4(params)
    dot_product = np.dot(n3, P4)
    return -dot_product


# 边界条件
bounds = [
    (1e-6, np.pi / 2 - 1e-6),  # theta
    (1e-6, np.pi - 1e-6),  # a0
    (1e-6, np.pi - 1e-6),  # a1
    (1e-6, np.pi - 1e-6),  # b0
    (1e-6, np.pi - 1e-6)  # b1
]

# 您提供的初始猜测值（需要满足 a0 < b0 < a1 < b1）
initial_guess = [np.pi / 8, 0.5, 2.0, 1.0, 2.5]  # 调整以满足约束


# 定义约束条件：a0 < b0 < a1 < b1
def constraint_a0_lt_b0(params):
    theta, a0, a1, b0, b1 = params
    return b0 - a0 - 1e-6  # b0 > a0 + 1e-6


def constraint_b0_lt_a1(params):
    theta, a0, a1, b0, b1 = params
    return a1 - b0 - 1e-6  # a1 > b0 + 1e-6


def constraint_a1_lt_b1(params):
    theta, a0, a1, b0, b1 = params
    return b1 - a1 - 1e-6  # b1 > a1 + 1e-6


constraints = [
    {'type': 'ineq', 'fun': constraint_a0_lt_b0},
    {'type': 'ineq', 'fun': constraint_b0_lt_a1},
    {'type': 'ineq', 'fun': constraint_a1_lt_b1}
]


def check_constraints(params):
    """检查参数是否满足约束"""
    theta, a0, a1, b0, b1 = params
    return a0 < b0 < a1 < b1


def run_optimization_with_constraints():
    """使用支持约束的优化方法"""

    methods = ['SLSQP', 'trust-constr']  # 只使用支持约束的方法
    results = {}

    print(f"目标最大违背值: {max_q_violation}")
    print(f"初始猜测的点积: {-objective(initial_guess)}")
    print(f"初始猜测是否满足约束: {check_constraints(initial_guess)}")

    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"正在使用 {method} 方法进行优化...")

        try:
            if method == 'SLSQP':
                options = {'maxiter': 5000, 'ftol': 1e-12, 'disp': False}
            elif method == 'trust-constr':
                options = {'maxiter': 5000, 'verbose': 0}

            result = minimize(objective, initial_guess, method=method,
                              bounds=bounds, constraints=constraints, options=options)

            results[method] = result

            # 计算最终的点积
            final_dot = -result.fun

            print(f"{method} 方法结果:")
            print(f"  最大点积: {final_dot:.8f}")
            print(f"  与目标差值: {abs(final_dot - max_q_violation):.8f}")
            print(f"  相对误差: {abs(final_dot - max_q_violation) / max_q_violation * 100:.4f}%")
            print(f"  迭代次数: {result.nit}")
            print(f"  函数调用次数: {result.nfev}")
            print(f"  是否收敛: {result.success}")
            print(f"  是否满足约束: {check_constraints(result.x)}")
            if result.message:
                print(f"  消息: {result.message}")

        except Exception as e:
            print(f"  {method} 方法出错: {e}")
            results[method] = None

    return results


def run_global_optimization_with_constraints():
    """使用全局优化方法（通过变量变换实现约束）"""
    print(f"\n{'=' * 60}")
    print("正在使用带约束的全局优化...")

    try:
        # 使用变量变换来确保 a0 < b0 < a1 < b1
        def transformed_objective(x):
            """
            使用变量变换确保 a0 < b0 < a1 < b1
            x = [theta, a0, delta1, delta2, delta3]
            其中: b0 = a0 + delta1, a1 = b0 + delta2, b1 = a1 + delta3
            """
            theta, a0, delta1, delta2, delta3 = x
            b0 = a0 + delta1
            a1 = b0 + delta2
            b1 = a1 + delta3
            return objective([theta, a0, a1, b0, b1])

        # 变换后的边界
        transformed_bounds = [
            (1e-6, np.pi / 2 - 1e-6),  # theta
            (1e-6, np.pi - 1e-6 - 3 * 1e-6),  # a0 (留出空间给deltas)
            (1e-6, np.pi - 1e-6),  # delta1
            (1e-6, np.pi - 1e-6),  # delta2
            (1e-6, np.pi - 1e-6)  # delta3
        ]

        # 变换初始猜测
        theta_guess, a0_guess, a1_guess, b0_guess, b1_guess = initial_guess
        delta1_guess = b0_guess - a0_guess
        delta2_guess = a1_guess - b0_guess
        delta3_guess = b1_guess - a1_guess
        transformed_guess = [theta_guess, a0_guess, delta1_guess, delta2_guess, delta3_guess]

        # 全局优化
        result_global = differential_evolution(
            transformed_objective,
            transformed_bounds,
            strategy='best1bin',
            popsize=25,
            maxiter=1500,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=True
        )

        # 转换回原始参数
        theta_opt, a0_opt, delta1_opt, delta2_opt, delta3_opt = result_global.x
        b0_opt = a0_opt + delta1_opt
        a1_opt = b0_opt + delta2_opt
        b1_opt = a1_opt + delta3_opt
        optimal_params = [theta_opt, a0_opt, a1_opt, b0_opt, b1_opt]

        # 局部精化
        result_refined = minimize(
            objective, optimal_params, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )

        final_dot = -result_refined.fun
        print("带约束的全局优化结果:")
        print(f"  最大点积: {final_dot:.8f}")
        print(f"  与目标差值: {abs(final_dot - max_q_violation):.8f}")
        print(f"  相对误差: {abs(final_dot - max_q_violation) / max_q_violation * 100:.4f}%")
        print(f"  是否满足约束: {check_constraints(result_refined.x)}")

        return result_refined

    except Exception as e:
        print(f"全局优化出错: {e}")
        return None


def multi_start_optimization_with_constraints(n_starts=10):
    """多起点优化（确保满足约束）"""
    print(f"\n{'=' * 60}")
    print(f"正在进行带约束的多起点优化 ({n_starts} 次尝试)...")

    best_result = None
    best_value = -np.inf

    for i in range(n_starts):
        print(f"尝试 {i + 1}/{n_starts}...")

        # 生成满足约束的随机猜测
        if i == 0:
            guess = initial_guess
        else:
            # 生成满足 a0 < b0 < a1 < b1 的随机点
            theta = np.random.uniform(1e-6, np.pi / 2 - 1e-6)
            a0 = np.random.uniform(1e-6, np.pi - 4 * 1e-6)
            b0 = np.random.uniform(a0 + 1e-6, np.pi - 3 * 1e-6)
            a1 = np.random.uniform(b0 + 1e-6, np.pi - 2 * 1e-6)
            b1 = np.random.uniform(a1 + 1e-6, np.pi - 1e-6)
            guess = [theta, a0, a1, b0, b1]

        try:
            result = minimize(
                objective,
                guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
            )

            if result.success and check_constraints(result.x):
                current_value = -result.fun
                if current_value > best_value:
                    best_value = current_value
                    best_result = result
                    print(f"  找到更好解: {current_value:.8f}")
            else:
                print(f"  解不满足约束或未收敛")

        except Exception as e:
            print(f"  尝试 {i + 1} 失败: {e}")

    if best_result:
        final_dot = -best_result.fun
        print("带约束的多起点优化最佳结果:")
        print(f"  最大点积: {final_dot:.8f}")
        print(f"  与目标差值: {abs(final_dot - max_q_violation):.8f}")
        print(f"  相对误差: {abs(final_dot - max_q_violation) / max_q_violation * 100:.4f}%")
        print(f"  是否满足约束: {check_constraints(best_result.x)}")

    return best_result


# 执行优化
if __name__ == "__main__":
    # 方法1: 使用支持约束的局部方法
    local_results = run_optimization_with_constraints()

    # 方法2: 带约束的全局优化
    global_result = run_global_optimization_with_constraints()

    # 方法3: 带约束的多起点优化
    multi_start_result = multi_start_optimization_with_constraints(10)

    # 收集所有有效结果
    all_results = {}
    for method, result in local_results.items():
        if result is not None and result.success and check_constraints(result.x):
            all_results[method] = result

    if global_result is not None and global_result.success and check_constraints(global_result.x):
        all_results['Global_Constrained'] = global_result

    if multi_start_result is not None and multi_start_result.success and check_constraints(multi_start_result.x):
        all_results['Multi_Start_Constrained'] = multi_start_result

    # 找到最佳结果
    if all_results:
        best_method = min(all_results.keys(),
                          key=lambda m: abs(-all_results[m].fun - max_q_violation))
        best_result = all_results[best_method]

        print(f"\n{'=' * 60}")
        print(f"🎯 最优方法: {best_method}")
        print(f"最优参数:")
        theta_opt, a0_opt, a1_opt, b0_opt, b1_opt = best_result.x
        print(f"  theta: {theta_opt:.8f} radians ({np.degrees(theta_opt):.6f} degrees)")
        print(f"  a0:    {a0_opt:.8f} radians ({np.degrees(a0_opt):.6f} degrees)")
        print(f"  a1:    {a1_opt:.8f} radians ({np.degrees(a1_opt):.6f} degrees)")
        print(f"  b0:    {b0_opt:.8f} radians ({np.degrees(b0_opt):.6f} degrees)")
        print(f"  b1:    {b1_opt:.8f} radians ({np.degrees(b1_opt):.6f} degrees)")

        # 验证约束条件
        print(f"\n约束验证:")
        print(f"  a0 < b0: {a0_opt:.6f} < {b0_opt:.6f} -> {a0_opt < b0_opt}")
        print(f"  b0 < a1: {b0_opt:.6f} < {a1_opt:.6f} -> {b0_opt < a1_opt}")
        print(f"  a1 < b1: {a1_opt:.6f} < {b1_opt:.6f} -> {a1_opt < b1_opt}")

        # 验证最终结果
        P4_optimal = compute_P4(best_result.x)
        final_dot_product = np.dot(n3, P4_optimal)
        print(f"\n最终点积: {final_dot_product:.8f}")
        print(f"目标最大违背值: {max_q_violation}")
        print(f"相对误差: {abs(final_dot_product - max_q_violation) / max_q_violation * 100:.6f}%")

    else:
        print("所有优化方法都失败了，或者没有找到满足约束的解")