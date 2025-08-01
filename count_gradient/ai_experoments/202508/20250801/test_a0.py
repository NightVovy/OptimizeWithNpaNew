import numpy as np

def find_a0(theta):
    right_side = np.arctan(np.sin(2 * theta))
    x = np.tan(theta) * np.tan(right_side / 2)
    a0 = 2 * np.arctan(x)
    return a0

# 扫描 theta 在 (0, pi/4) 之间的值
theta_values = np.linspace(0.001, np.pi/4 - 0.001, 1000)  # 避免边界
a0_values = [find_a0(theta) for theta in theta_values]

# 找到 a0 的最大值及其对应的 theta
max_a0 = max(a0_values)
max_index = np.argmax(a0_values)
theta_max = theta_values[max_index]

print(f"在 theta ∈ (0, π/4) 时，a0 的最大值为 {max_a0:.4f} (≈ {np.degrees(max_a0):.2f}°)")
print(f"对应的 theta = {theta_max:.4f} (≈ {np.degrees(theta_max):.2f}°)")

# 绘制 a0 随 theta 的变化
import matplotlib.pyplot as plt
plt.plot(theta_values, a0_values, label='a0 vs theta')
plt.axvline(x=theta_max, color='r', linestyle='--', label=f'max a0 at theta={theta_max:.4f}')
plt.xlabel('theta (radians)')
plt.ylabel('a0 (radians)')
plt.title('a0 as a function of theta in (0, π/4)')
plt.legend()
plt.grid()
plt.show()