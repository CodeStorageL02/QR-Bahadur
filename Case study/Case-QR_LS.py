import numpy as np
import pandas as pd
from related_functions import estimate_sq, build_omega,build_R
import matplotlib.pyplot as plt
from IRLS import IRLS
import random

DF1 = pd.read_excel('train.xlsx')
DF2 = pd.read_excel('test.xlsx')
ones_1 = np.ones((DF1.shape[0],1))
ones_2 = np.ones((DF2.shape[0],1))
DF1['intercept'] = ones_1
DF2['intercept'] = ones_2
print(DF1.shape)
print(DF2.shape)
Day_tra = int(DF1.shape[0]/78)
print(Day_tra)


p=5
covariates = ['intercept','x1','x2','x3','x4']

y_f1 = DF1['y'].to_numpy().reshape(-1,1)

x_f1 = DF1[covariates].to_numpy()

print(x_f1.shape)

# IC 状态参数估计
tau = 0.5
rho = 0.25
lam = 0.2



outliers = [22, 23, 24, 25, 29, 30, 31, 33, 34,35]  # 1sigma
#
DF1 = DF1[~DF1['Day_number'].isin(outliers)]
#Bootstrap for control limit
D_l = DF1['Day_number'].unique()


random.seed(2)
RLT_Sum_list=[]
for i in range(1000):
    print(i)
    Rlt_I = []

    training_VN = random.sample(list(D_l),int(len(D_l)*0.4))
    test_VN = [v for v in D_l if v not in training_VN]
    # training
    training_da = DF1[DF1['Day_number'].isin(training_VN)]
    test_da = DF1[DF1['Day_number'].isin(test_VN)]
    x_tra = training_da[covariates].to_numpy()
    y_tra = training_da['y'].to_numpy().reshape(-1,1)

    beta_tau_0 = IRLS(tau, y_tra, x_tra)
    beta_rho_0 = IRLS(rho, y_tra, x_tra)
    beta_1rho_0 = IRLS(1 - rho, y_tra, x_tra)

    D_hat = x_tra.T @ x_tra / x_tra.shape[0]

    s_tau = estimate_sq(x_tra, y_tra, tau)
    s_rho = estimate_sq(x_tra, y_tra, rho)
    s_1rho = estimate_sq(x_tra, y_tra, 1 - rho)

    be_all_0 = np.concatenate([beta_tau_0, beta_rho_0, beta_1rho_0])

    Omega = build_omega([tau, rho, 1 - rho], [s_tau, s_rho, s_1rho])

    R = build_R(p)
    Sigma = R @ np.kron(Omega, np.linalg.inv(D_hat)) @ R.T
    S_inv = np.linalg.inv(Sigma)


    # test
    for j in test_VN:
        data = test_da[test_da['Day_number'] == j]
        xi = data[covariates].to_numpy()
        yi = data['y'].to_numpy().reshape(-1,1)
        n = xi.shape[0]

        beta_tau = IRLS(tau, yi, xi)
        beta_rho = IRLS(rho, yi, xi)
        beta_1rho = IRLS(1 - rho, yi, xi)
        det = np.concatenate([beta_tau, beta_rho, beta_1rho]) - be_all_0
        v = R @ det
        z = (1 - lam) * z + lam * v

        Rlt = n * np.squeeze(z) @ S_inv @ np.squeeze(z)


        Rlt_I.append(Rlt)
    RLT_Sum_list.extend(Rlt_I)

RLT_Sum_list = np.sort(RLT_Sum_list)[::-1]
limit = RLT_Sum_list[int(len(RLT_Sum_list) * 0.005)]
print(limit)


# phase II

y_f1 = DF1['y'].to_numpy().reshape(-1,1)

x_f1 = DF1[covariates].to_numpy()

beta_tau_0 = IRLS(tau, y_f1, x_f1)
beta_rho_0 = IRLS(rho, y_f1, x_f1)
beta_1rho_0 = IRLS(1 - rho,y_f1, x_f1)

# 估计设计矩阵

D_hat = x_f1.T @ x_f1 / x_f1.shape[0]
print(D_hat)

# 估计稀疏函数 s(q)
s_tau = estimate_sq(x_f1,y_f1, tau)
s_rho = estimate_sq(x_f1,y_f1, rho)
s_1rho = estimate_sq(x_f1,y_f1, 1 - rho)
print(s_tau,s_rho,s_1rho)

be_all_0 = np.concatenate([beta_tau_0, beta_rho_0, beta_1rho_0])
print(be_all_0)

# 构建协方差矩阵
Omega = build_omega([tau, rho, 1 - rho], [s_tau, s_rho, s_1rho])

R = build_R(p)
Sigma = R @ np.kron(Omega, np.linalg.inv(D_hat)) @ R.T
S_inv = np.linalg.inv(Sigma)




P2_list = DF2['Day_number'].unique()
RLT_II =[]
z = np.zeros((2 * p, 1))

for i in P2_list:
    data = DF2[DF2['Day_number'] == i]
    x2 = data[covariates].to_numpy()
    y2 = data['y'].to_numpy().reshape(-1,1)
    n = x2.shape[0]

    beta_tau = IRLS(tau, y2, x2)
    beta_rho = IRLS(rho, y2, x2)
    beta_1rho = IRLS(1 - rho, y2, x2)
    det = np.concatenate([beta_tau, beta_rho, beta_1rho]) - be_all_0
    v = R @ det
    z = (1 - lam) * z + lam * v

    Rlt = n * np.squeeze(z) @ S_inv @ np.squeeze(z)
    RLT_II.append(Rlt)
print(RLT_II)
non_normal=[]
for i in range(Day_tra):
    if RLT_II[i] >= limit:
        non_normal.append(i)
print(non_normal)

T = len(P2_list)
plt.figure(figsize=(16,4))
plt.plot([i+1 for i in range(T)], np.log(RLT_II), 'b-',marker='.',label='$\ln$(QR-LS statistic)')
plt.plot([i+1 for i in range(T)], [np.log(limit) for i in range(T)], 'r-', label='$\ln$(control limit)')
plt.tick_params(axis='x', direction='inout')
plt.xlim(0,T+1)
#plt.xticks([i+1 for i in range(T)])
plt.xlabel('Day Number')
plt.legend()
#plt.savefig('QR-LS_.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
#plt.title(j)
plt.show()
