import sympy as sp

# 初始化符号和常量
theta_val = sp.pi / 6  # θ = π/6
a0_val = -sp.atan(sp.sin(2 * theta_val))
a1_val = sp.atan(sp.sin(2 * theta_val))
b0, b1 = sp.symbols('b0 b1')

# 打印初始值
print("=== 固定参数值 ===")
print(f"θ = {theta_val:.6f} radians (π/6)")
print(f"固定 a0 = -arctan(sin(2θ)) = {a0_val:.6f}")
print(f"固定 a1 = arctan(sin(2θ)) = {a1_val:.6f}\n")


# 定义角度变换函数（固定a0,a1,theta，只计算变换后的值）
def calculate_transforms():
    transforms = {
        'a0_plus': 2 * sp.atan(sp.tan(a0_val / 2) * sp.tan(theta_val)),
        'a0_minus': 2 * sp.atan(sp.tan(a0_val / 2) / sp.tan(theta_val)),
        'a1_plus': 2 * sp.atan(sp.tan(a1_val / 2) * sp.tan(theta_val)),
        'a1_minus': 2 * sp.atan(sp.tan(a1_val / 2) / sp.tan(theta_val)),
    }
    return {k: v.evalf() for k, v in transforms.items()}


# 计算固定变换值
transforms = calculate_transforms()

# 问题1：解a0_plus = b0
print("=== 问题1：解a0_plus = b0 ===")
eq1 = transforms['a0_plus'] - b0
sol_b0_1 = sp.solve(eq1, b0)[0]
print(f"解: b0 = {sol_b0_1:.6f}")

# 问题2：解b0 = a1_plus
print("\n=== 问题2：解b0 = a1_plus ===")
eq2 = b0 - transforms['a1_plus']
sol_b0_2 = sp.solve(eq2, b0)[0]
print(f"解: b0 = {sol_b0_2:.6f}")

# 问题3：解a1_minus = b1
print("\n=== 问题3：解a1_minus = b1 ===")
eq3 = transforms['a1_minus'] - b1
sol_b1 = sp.solve(eq3, b1)[0]
print(f"解: b1 = {sol_b1:.6f}")


# 验证不等式链
def verify_inequality(b0_val, b1_val):
    values = [
        float(transforms['a0_minus']),
        float(transforms['a0_plus']),
        float(b0_val),
        float(transforms['a1_plus']),
        float(transforms['a1_minus']),
        float(b1_val)
    ]

    tol = 1e-10
    checks = [
        (values[1] - values[0]) >= -tol,
        (values[2] - values[1]) >= -tol,
        (values[3] - values[2]) >= -tol,
        (values[4] - values[3]) >= -tol,
        (values[5] - values[4]) >= -tol
    ]
    return all(checks), values


# 验证各个解
print("\n=== 验证结果 ===")
valid1, chain1 = verify_inequality(sol_b0_1, sol_b1)
print(f"问题1解的不等式验证: {'通过' if valid1 else '失败'}")
print(
    f"   {chain1[0]:.6f} <= {chain1[1]:.6f} <= {chain1[2]:.6f} <= {chain1[3]:.6f} <= {chain1[4]:.6f} <= {chain1[5]:.6f}")

valid2, chain2 = verify_inequality(sol_b0_2, sol_b1)
print(f"\n问题2解的不等式验证: {'通过' if valid2 else '失败'}")
print(
    f"   {chain2[0]:.6f} <= {chain2[1]:.6f} <= {chain2[2]:.6f} <= {chain2[3]:.6f} <= {chain2[4]:.6f} <= {chain2[5]:.6f}")

valid3, chain3 = verify_inequality(sol_b0_1, sol_b1)
print(f"\n问题3解的不等式验证: {'通过' if valid3 else '失败'}")
print(
    f"   {chain3[0]:.6f} <= {chain3[1]:.6f} <= {chain3[2]:.6f} <= {chain3[3]:.6f} <= {chain3[4]:.6f} <= {chain3[5]:.6f}")


# 输出详细角度值
def print_angles(b0_val, b1_val, title):
    print(f"\n{title}:")
    print(f"   a0_minus: {float(transforms['a0_minus']):.6f}")
    print(f"   a0_plus: {float(transforms['a0_plus']):.6f}")
    print(f"   b0: {float(b0_val):.6f}")
    print(f"   a1_plus: {float(transforms['a1_plus']):.6f}")
    print(f"   a1_minus: {float(transforms['a1_minus']):.6f}")
    print(f"   b1: {float(b1_val):.6f}")


print_angles(sol_b0_1, sol_b1, "问题1详细角度值")
print_angles(sol_b0_2, sol_b1, "问题2详细角度值")
print_angles(sol_b0_1, sol_b1, "问题3详细角度值")