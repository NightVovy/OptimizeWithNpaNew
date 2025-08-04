import sympy as sp

# 初始化符号和常量
theta_val = sp.pi/6  # θ = π/6
a0, a1, theta = sp.symbols('a0 a1 theta')
b0 = 0
b1 = sp.pi/2

# 打印初始值
print("=== 初始参数值 ===")
print(f"θ = {theta_val:.6f} radians (π/6)")
print(f"初始 a0 = -arctan(sin(2θ)) = {-sp.atan(sp.sin(2*theta_val)):.6f}")
print(f"初始 a1 = arctan(sin(2θ)) = {sp.atan(sp.sin(2*theta_val)):.6f}")
print(f"b0 = 0 = {b0:.6f} radians")
print(f"b1 = π/2 = {b1:.6f} radians\n")

# 定义角度变换函数
def angle_transforms(a0_val, a1_val):
    transforms = {
        'a0_plus': 2 * sp.atan(sp.tan(a0_val / 2) * sp.tan(theta)),
        'a0_minus': 2 * sp.atan(sp.tan(a0_val / 2) / sp.tan(theta)),
        'a1_plus': 2 * sp.atan(sp.tan(a1_val / 2) * sp.tan(theta)),
        'a1_minus': 2 * sp.atan(sp.tan(a1_val / 2) / sp.tan(theta)),
        'b0': b0,  # 直接使用数值，无需替换
        'b1': b1  # 直接使用数值，无需替换
    }

    # 仅对需要替换 theta 的项调用 .subs()
    result = {}
    for k, v in transforms.items():
        if k in ['a0_plus', 'a0_minus', 'a1_plus', 'a1_minus']:
            result[k] = v.subs(theta, theta_val)
        else:
            result[k] = v  # b0 和 b1 直接保留
    return result

# 验证不等式链
def verify_inequality(angles):
    # 提取值，仅对符号表达式调用 .evalf()
    values = [
        float(angles['a0_minus'].evalf()) if hasattr(angles['a0_minus'], 'evalf') else angles['a0_minus'],
        float(angles['a0_plus'].evalf()) if hasattr(angles['a0_plus'], 'evalf') else angles['a0_plus'],
        float(angles['b0']),  # b0 是普通数值，直接使用
        float(angles['a1_plus'].evalf()) if hasattr(angles['a1_plus'], 'evalf') else angles['a1_plus'],
        float(angles['a1_minus'].evalf()) if hasattr(angles['a1_minus'], 'evalf') else angles['a1_minus'],
        float(angles['b1'])   # b1 是普通数值，直接使用
    ]

    # 使用带容差的比较
    tol = 1e-10
    checks = [
        (values[1] - values[0]) >= -tol,
        (values[2] - values[1]) >= -tol,
        (values[3] - values[2]) >= -tol,
        (values[4] - values[3]) >= -tol,
        (values[5] - values[4]) >= -tol  # 有修改
    ]
    return all(checks), values

# 问题1：解a0_plus = b0 (固定a1=arctan(sin(2θ)))
print("=== 问题1：解a0_plus = b0 ===")
a1_init = sp.atan(sp.sin(2*theta_val))
angles_init = angle_transforms(a0, a1_init)
eq_a0 = angles_init['a0_plus'] - angles_init['b0']
sol_a0 = sp.solve(eq_a0, a0)
a0_sol = [sol.evalf() for sol in sol_a0 if sol.is_real][0]

# 问题2：解a1_plus = b0 (固定a0=-arctan(sin(2θ)))
print("\n=== 问题2：解a1_plus = b0 ===")
a0_init = -sp.atan(sp.sin(2*theta_val))
angles_init = angle_transforms(a0_init, a1)
eq_a1 = angles_init['a1_plus'] - angles_init['b0']
sol_a1 = sp.solve(eq_a1, a1)
a1_sol1 = [sol.evalf() for sol in sol_a1 if sol.is_real][0]

# 问题3：解a1_minus = b1 (固定a0=-arctan(sin(2θ)))
print("\n=== 问题3：解a1_minus = b1 ===")
angles_init = angle_transforms(a0_init, a1)
eq_a1m = angles_init['a1_minus'] - angles_init['b1']
sol_a1m = sp.solve(eq_a1m, a1)
real_sols = [sol.evalf() for sol in sol_a1m if sol.is_real]
a1_sol2 = real_sols[0] if real_sols else None

# 输出最终结果
print("\n=== 三个方程的解 ===")
print(f"1. a0_plus = b0 的解: a0 = {a0_sol:.6f} radians")
angles_ver1 = angle_transforms(a0_sol, a1_init)
valid1, chain1 = verify_inequality(angles_ver1)
print("   不等式验证值:")
print(f"   {chain1[0]:.6f} <= {chain1[1]:.6f} <= {chain1[2]:.6f} <= {chain1[3]:.6f} <= {chain1[4]:.6f} <= {chain1[5]:.6f}")
print(f"   验证结果: {'通过' if valid1 else '失败'}")

print(f"\n2. a1_plus = b0 的解: a1 = {a1_sol1:.6f} radians")
angles_ver2 = angle_transforms(a0_init, a1_sol1)
valid2, chain2 = verify_inequality(angles_ver2)
print("   不等式验证值:")
print(f"   {chain2[0]:.6f} <= {chain2[1]:.6f} <= {chain2[2]:.6f} <= {chain2[3]:.6f} <= {chain2[4]:.6f} <= {chain2[5]:.6f}")
print(f"   验证结果: {'通过' if valid2 else '失败'}")

if a1_sol2 is not None:
    print(f"\n3. a1_minus = b1 的解: a1 = {a1_sol2:.6f} radians")
    angles_ver3 = angle_transforms(a0_init, a1_sol2)
    valid3, chain3 = verify_inequality(angles_ver3)
    print("   不等式验证值:")
    print(f"   {chain3[0]:.6f} <= {chain3[1]:.6f} <= {chain3[2]:.6f} <= {chain3[3]:.6f} <= {chain3[4]:.6f} <= {chain3[5]:.6f}")
    print(f"   验证结果: {'通过' if valid3 else '失败'}")
else:
    print("\n3. a1_minus = b1 无有效解")

# 详细角度值输出（按要求的顺序）
def print_angles(angles, title):
    print(f"\n{title}:")
    print(f"   a0_minus: {float(angles['a0_minus']):.6f}")
    print(f"   a0_plus: {float(angles['a0_plus']):.6f}")
    print(f"   b0: {float(angles['b0']):.6f}")
    print(f"   a1_plus: {float(angles['a1_plus']):.6f}")
    print(f"   a1_minus: {float(angles['a1_minus']):.6f}")
    print(f"   b1: {float(angles['b1']):.6f}")

print_angles(angles_ver1, "问题1详细角度值")
print_angles(angles_ver2, "问题2详细角度值")
if a1_sol2:
    print_angles(angles_ver3, "问题3详细角度值")