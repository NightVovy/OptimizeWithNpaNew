import sympy as sp

# Define variables
a0, a1, theta = sp.symbols('a0 a1 theta')

# Given values
theta_value = sp.pi / 6
a0_initial = 0
a1_initial = sp.pi / 2
b0 = sp.atan(sp.sin(2 * theta_value))
b1 = (3 * sp.pi / 2) + b0


# Definitions
a0_plus = 2 * sp.atan(sp.tan(a0 / 2) * sp.tan(theta))
a0_minus = 2 * sp.atan(sp.tan(a0 / 2) / sp.tan(theta))
a1_plus = 2 * sp.atan(sp.tan(a1 / 2) * sp.tan(theta))
a1_minus = 2 * sp.atan(sp.tan(a1 / 2) / sp.tan(theta))

# Initial values
print("Initial values:")
print(f"a0_plus = {a0_plus.subs({a0: a0_initial, theta: theta_value}).evalf()}")
print(f"a0_minus = {a0_minus.subs({a0: a0_initial, theta: theta_value}).evalf()}")
print(f"b0 = {b0.evalf()}")
print(f"a1_plus = {a1_plus.subs({a1: a1_initial, theta: theta_value}).evalf()}")
print(f"a1_minus = {a1_minus.subs({a1: a1_initial, theta: theta_value}).evalf()}")
print(f"b1 = {b1.evalf()}")

# Solve a0_minus = b0
equation_a0 = a0_minus.subs(theta, theta_value) - b0
solution_a0 = sp.solve(equation_a0, a0)
a0_solution = solution_a0[0]  # Assuming one solution
print(f"\nSolution for a0: {a0_solution.evalf()} radians")

# Solve a1_plus = b0
equation_a1 = a1_plus.subs(theta, theta_value) - b0
solution_a1 = sp.solve(equation_a1, a1)
a1_solution = solution_a1[0]  # Assuming one solution
print(f"Solution for a1: {a1_solution.evalf()} radians")

# Verify inequalities with new a0 and a1
a0_new = a0_solution.evalf()
a1_new = a1_solution.evalf()

a0_plus_new = a0_plus.subs({a0: a0_new, theta: theta_value}).evalf()
a0_minus_new = a0_minus.subs({a0: a0_new, theta: theta_value}).evalf()
a1_plus_new = a1_plus.subs({a1: a1_new, theta: theta_value}).evalf()
a1_minus_new = a1_minus.subs({a1: a1_new, theta: theta_value}).evalf()

print("\nVerification with new a0 and a1:")
print(f"a0_plus = {a0_plus_new}")
print(f"a0_minus = {a0_minus_new}")
print(f"b0 = {b0.evalf()}")
print(f"a1_plus = {a1_plus_new}")
print(f"a1_minus = {a1_minus_new}")
print(f"b1 = {b1.evalf()}")
print(f"Inequality: 0 < {a0_plus_new} <= {a0_minus_new} <= {b0.evalf()} <= {a1_plus_new} <= {a1_minus_new} < {b1.evalf()}")
print(f"Inequality holds: {a0_plus_new > 0 and a0_plus_new <= a0_minus_new <= b0.evalf() <= a1_plus_new <= a1_minus_new < b1.evalf()}")