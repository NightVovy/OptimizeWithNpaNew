这个文件夹只存alpha不等式的验证。
alphaA0+A0B0-A0B1+A1B0+A1B1<= sqrt(8+2*alpha^2)
测量设置在
    A0 = np.cos(a0) * Z + np.sin(a0) * X
    A1 = np.cos(a1) * Z + np.sin(a1) * X
    B0 = np.cos(b0) * Z + np.sin(b0) * X
    B1 = - np.cos(b1) * Z + np.sin(b1) * X
的时候达到最大违背。