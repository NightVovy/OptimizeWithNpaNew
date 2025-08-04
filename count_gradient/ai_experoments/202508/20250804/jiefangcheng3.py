import sympy as sp

# 初始化符号和常量
theta_val = sp.pi/6  # θ = π/6
a0_val, a1_val, theta, b0, b1 = sp.symbols('a0_val a1_val theta b0 b1')

# 定义初始值
a0_initial = 0  # 初始a0值
a1_initial = sp.pi/2  # 初始a1值
b0_initial = sp.atan(sp.sin(2*theta_val))  # b0初始定义
b1_initial = (3*sp.pi/2) + b0_initial  # b1初始定义

# 计算初始值的数值结果
b0_initial_num = float(b0_initial.evalf())
b1_initial_num = float(b1_initial.evalf())

# 打印初始值
print("=== 初始参数值 ===")
print(f"θ = {theta_val:.6f} radians (π/6)")
print(f"a0 = {a0_initial:.6f}")
print(f"a1 = π/2 = {float(a1_initial.evalf()):.6f}")
print(f"b0初始定义 = arctan(sin(2θ)) = {b0_initial_num:.6f} radians")
print(f"b1初始定义 = 3π/2 + b0 = {b1_initial_num:.6f} radians")
print()

# 定义角度变换函数
def angle_transforms(a0_val, a1_val, b0_val, b1_val):
    transforms = {
        'a0_plus': 2 * sp.atan(sp.tan(a0_val/2) * sp.tan(theta)),
        'a0_minus': 2 * sp.atan(sp.tan(a0_val/2) / sp.tan(theta)),
        'a1_plus': 2 * sp.atan(sp.tan(a1_val/2) * sp.tan(theta)),
        'a1_minus': 2 * sp.atan(sp.tan(a1_val/2) / sp.tan(theta)),
        'b0': b0_val,
        'b1': b1_val
    }
    return {k: v.subs(theta, theta_val) for k, v in transforms.items()}

# 验证不等式链
def verify_inequality(angles):
    values = [
        float(angles['a0_plus'].evalf()),
        float(angles['a0_minus'].evalf()),
        float(angles['b0'].evalf()),
        float(angles['a1_plus'].evalf()),
        float(angles['a1_minus'].evalf()),
        float(angles['b1'].evalf())
    ]

    # 使用带容差的比较
    tol = 1e-10  # 允许的误差范围

    checks = [
        values[0] >= -tol,
        (values[1] - values[0]) >= -tol,
        (values[2] - values[1]) >= -tol,
        (values[3] - values[2]) >= -tol,
        (values[4] - values[3]) >= -tol,
        (values[5] - values[4]) >= -tol
    ]
    return all(checks), values

# 问题1：解a0_minus = b0 (使用初始a0和a1值)
print("=== 问题1：解a0_minus = b0 ===")
angles_init = angle_transforms(a0_initial, a1_initial, b0, b1)
eq_b0 = angles_init['a0_minus'] - angles_init['b0']
sol_b0 = sp.solve(eq_b0, b0)
b0_sol = [sol.evalf() for sol in sol_b0 if sol.is_real and sol.evalf() >= 0][0]
b0_sol_num = float(b0_sol)  # 转换为数值

# 问题2：解a1_plus = b0 (使用初始a0和a1值)
print("\n=== 问题2：解a1_plus = b0 ===")
angles_init = angle_transforms(a0_initial, a1_initial, b0, b1)
eq_b0_2 = angles_init['a1_plus'] - angles_init['b0']
sol_b0_2 = sp.solve(eq_b0_2, b0)
b0_sol2 = [sol.evalf() for sol in sol_b0_2 if sol.is_real and sol.evalf() >= 0][0]
b0_sol2_num = float(b0_sol2)  # 转换为数值

# 问题3：解a1_minus = b1 (使用初始a0和a1值)
print("\n=== 问题3：解a1_minus = b1 ===")
angles_init = angle_transforms(a0_initial, a1_initial, b0, b1)
eq_b1 = angles_init['a1_minus'] - angles_init['b1']
sol_b1 = sp.solve(eq_b1, b1)
real_sols = [sol.evalf() for sol in sol_b1 if sol.is_real and sol.evalf() >= 0]
b1_sol = real_sols[0] if real_sols else None
b1_sol_num = float(b1_sol) if b1_sol else None  # 转换为数值

# 输出最终结果
print("\n=== 三个方程的解 ===")
print(f"1. a0_minus = b0 的解: b0 = {b0_sol_num:.6f} radians")
print(f"   与初始b0的差异: {abs(b0_sol_num - b0_initial_num):.6e}")
angles_ver1 = angle_transforms(a0_initial, a1_initial, b0_sol, b1_initial)
valid1, chain1 = verify_inequality(angles_ver1)
print("   不等式验证值:")
print(f"   0 <= {chain1[0]:.6f} <= {chain1[1]:.6f} <= {chain1[2]:.6f} <= {chain1[3]:.6f} <= {chain1[4]:.6f} <= {chain1[5]:.6f}")
print(f"   验证结果: {'通过' if valid1 else '失败'}")

print(f"\n2. a1_plus = b0 的解: b0 = {b0_sol2_num:.6f} radians")
print(f"   与初始b0的差异: {abs(b0_sol2_num - b0_initial_num):.6e}")
angles_ver2 = angle_transforms(a0_initial, a1_initial, b0_sol2, b1_initial)
valid2, chain2 = verify_inequality(angles_ver2)
print("   不等式验证值:")
print(f"   0 <= {chain2[0]:.6f} <= {chain2[1]:.6f} <= {chain2[2]:.6f} <= {chain2[3]:.6f} <= {chain2[4]:.6f} <= {chain2[5]:.6f}")
print(f"   验证结果: {'通过' if valid2 else '失败'}")

if b1_sol is not None:
    print(f"\n3. a1_minus = b1 的解: b1 = {b1_sol_num:.6f} radians")
    print(f"   与初始b1的差异: {abs(b1_sol_num - b1_initial_num):.6e}")
    angles_ver3 = angle_transforms(a0_initial, a1_initial, b0_initial, b1_sol)
    valid3, chain3 = verify_inequality(angles_ver3)
    print("   不等式验证值:")
    print(f"   0 <= {chain3[0]:.6f} <= {chain3[1]:.6f} <= {chain3[2]:.6f} <= {chain3[3]:.6f} <= {chain3[4]:.6f} <= {chain3[5]:.6f}")
    print(f"   验证结果: {'通过' if valid3 else '失败'}")
else:
    print("\n3. a1_minus = b1 无有效解")

# 详细角度值输出
def print_angles(angles, title):
    print(f"\n{title}:")
    print(f"   a0_plus: {float(angles['a0_plus'].evalf()):.6f}")
    print(f"   a0_minus: {float(angles['a0_minus'].evalf()):.6f}")
    print(f"   b0: {float(angles['b0'].evalf()):.6f}")
    print(f"   a1_plus: {float(angles['a1_plus'].evalf()):.6f}")
    print(f"   a1_minus: {float(angles['a1_minus'].evalf()):.6f}")
    print(f"   b1: {float(angles['b1'].evalf()):.6f}")

print_angles(angles_ver1, "问题1详细角度值")
print_angles(angles_ver2, "问题2详细角度值")
if b1_sol:
    print_angles(angles_ver3, "问题3详细角度值")