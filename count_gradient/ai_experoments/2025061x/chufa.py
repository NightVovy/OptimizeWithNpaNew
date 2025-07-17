from math import sqrt

result1 = sqrt(2) / sqrt(2)
result2 = 1.41421356 / 1.41421356

A0 = 0
A1 = 0
B0 = 0
B1 = 0
E00 = 0.707107
E01 = -0.707107
E10 = 0.707107
E11 = 0.707107
s=1
t=1

def new_corre(Exy, Ax, By, s):
    return (Exy + s * By) / (1 + s * Ax)

nc1 = new_corre(E00, A0, B0, s)
nc2 = new_corre(E01, A0, B1, s)
nc3 = new_corre(E10, A1, B0, t)
nc4 = new_corre(E11, A1, B1, t)



term1 = - (s * (E00 + s * B0)) / ((1 + s * A0)**2 * sqrt(1 - nc1**2)) \
            + (s * (E01 + s * B1)) / ((1 + s * A0)**2 * sqrt(1 - nc2**2))


print(result1)  # 通常输出 1.0
print(result2)  # 通常输出 1.0
print(term1)
