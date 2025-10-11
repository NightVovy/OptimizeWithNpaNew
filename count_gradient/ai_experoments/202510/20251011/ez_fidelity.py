# 定义两个保真度值
fidelity_origin = 0.9999999821597665
fidelity_new = 0.9999999958262309

# 比较大小
if fidelity_new > fidelity_origin:
    print(f"fidelity_new ({fidelity_new}) 大于 fidelity_origin ({fidelity_origin})")
elif fidelity_new < fidelity_origin:
    print(f"fidelity_new ({fidelity_new}) 小于 fidelity_origin ({fidelity_origin})")
else:
    print(f"fidelity_new ({fidelity_new}) 等于 fidelity_origin ({fidelity_origin})")
