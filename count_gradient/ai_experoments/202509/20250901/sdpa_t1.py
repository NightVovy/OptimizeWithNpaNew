from ncpol2sdpa import SdpRelaxation, Probability, define_objective_with_I, maximum_violation

# level = 1 + AB relaxation for CHSH
level = 1
I = [[0, -1, 0], [-1, 1, 1], [0, 1, -1]]

# 或使用更底层方式构造 SDP:
P = Probability([2,2],[2,2])

objective1 = define_objective_with_I(I, P)

CHSH1 = -P([0],[0],'A') + P([0,0],[0,0])  + P([0,0],[0,1]) + \
        P([0,0],[1,0]) - P([0,0],[1,1]) - P([0],[0],'B')
CHSH2 =  P([0,0],[0,0])  + P([0,0],[0,1]) + \
        P([0,0],[1,0]) - P([0,0],[1,1])
objective2 = -CHSH2


sdp = SdpRelaxation(P.get_all_operators())
sdp.get_relaxation(level, objective=objective2,
                   substitutions=P.substitutions,
                   extramonomials=P.get_extra_monomials('AB'))
sdp.solve(solver="mosek")
print(sdp.primal)