def cal_tot(exp):
    tot = 0
    for item in exp:
        tot = tot+item
    return tot

arsh_exp = [2340, 2500, 2100, 3100, 2980]
god_exp = [200, 500, 700]

arsh_tot=cal_tot(arsh_exp)
god_tot=cal_tot(god_exp)
print('Arsh total expense is: ', arsh_tot)
print('God total expense is: ', god_tot)