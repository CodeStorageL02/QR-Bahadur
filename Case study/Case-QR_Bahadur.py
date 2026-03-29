import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from IRLS import IRLS_List, generate_Omega
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

#mean0 = np.mean(x_f1,axis=0)
#std0 = np.std(x_f1,axis=0)

#x_f1 = x_f1 - mean0
# IC 状态参数估计
tau_list = [0.1, 0.5,  0.9]
Cov = x_f1.T @ x_f1/x_f1.shape[0]
Omega = generate_Omega(p,tau_list,Cov)
print(Omega.shape)
Omega_inv = np.linalg.inv(Omega)

Beta_0 = IRLS_List(y_f1,x_f1,tau_list)
print(Beta_0)
lam = 0.2



# outliers
Rlt_I =[]
w = np.zeros((len(tau_list), 5))
P1_list = DF1['Day_number'].unique()
for i in P1_list:
    data = DF1[DF1['Day_number'] == i]
    xi = data[covariates].to_numpy()
    yi = data['y'].to_numpy().reshape(-1,1)
    n = xi.shape[0]
    for a in range(len(tau_list)):
        res = np.squeeze(yi) - xi @ Beta_0[:, a]
        resid = np.where(res < 0, tau_list[a] - 1, tau_list[a])
        z = np.sum(xi.T * resid, axis=1) / n
        w[a, :] = (1 - lam) * w[a, :] + lam * z
    w_vec = w.flatten('C')
    Rlt = n * w_vec @ Omega_inv @ w_vec.T

    Rlt_I.append(Rlt)
print(Rlt_I)
plt.figure(figsize=(16,4))
plt.plot([i+1 for i in range(len(Rlt_I))], np.log(Rlt_I), 'b-',marker='*',label='$\ln$(QR-BR statistic)') #'b.'
plt.xlabel('Day Number')
plt.legend()
plt.show()

print(DF1.shape)


outliers=[]
for i in range(len(Rlt_I)):
    if Rlt_I[i] >= np.mean(Rlt_I)+np.std(Rlt_I):
        outliers.append(i)
print(outliers)

with open("Day.txt", "r") as file:
    S_D = file.read().splitlines()
print(len(S_D))
S_D = np.array(S_D)
print(S_D[outliers])


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
    cov_tra = x_f1.T @ x_f1 / x_tra.shape[0]
    Omega = generate_Omega(p, tau_list, cov_tra)
    Omega_inv_tra = np.linalg.inv(Omega)
    Beta_0 = IRLS_List(y_tra, x_tra, tau_list)

    # test
    w = np.zeros((len(tau_list), 5))
    for j in test_VN:
        data = test_da[test_da['Day_number'] == j]
        xi = data[covariates].to_numpy()
        yi = data['y'].to_numpy().reshape(-1,1)
        n = xi.shape[0]
        for a in range(len(tau_list)):
            res = np.squeeze(yi) - xi @ Beta_0[:, a]
            resid = np.where(res < 0, tau_list[a] - 1, tau_list[a])
            z = np.sum(xi.T * resid, axis=1) / n
            w[a, :] = (1 - lam) * w[a, :] + lam * z
        w_vec = w.flatten('C')
        Rlt = n * w_vec @ Omega_inv_tra @ w_vec.T
        Rlt_I.append(Rlt)
    RLT_Sum_list.extend(Rlt_I)

RLT_Sum_list = np.sort(RLT_Sum_list)[::-1]
limit = RLT_Sum_list[int(len(RLT_Sum_list) * 0.005)]
print(limit)


# phase II

y_f1 = DF1['y'].to_numpy().reshape(-1,1)

x_f1 = DF1[covariates].to_numpy()

M = x_f1.T @ x_f1 / x_f1.shape[0]
Omega = generate_Omega(p,tau_list,M)
print(Omega.shape)
Omega_inv_2 = np.linalg.inv(Omega)

# IC 状态参数估计
#tau_list = [0.1, 0.5, 0.9]

Beta_0 = IRLS_List(y_f1,x_f1,tau_list)
print(Beta_0)
print('************************')
P2_list = DF2['Day_number'].unique()
RLT_II =[]
w = np.zeros((len(tau_list), 5))
for i in P2_list:
    data = DF2[DF2['Day_number'] == i]
    x2 = data[covariates].to_numpy()
    y2 = data['y'].to_numpy().reshape(-1,1)
    n = x2.shape[0]

    for a in range(len(tau_list)):
        res = np.squeeze(y2) - x2 @ Beta_0[:, a]
        resid = np.where(res < 0, tau_list[a] - 1, tau_list[a])
        z = np.sum(x2.T * resid, axis=1) / n
        w[a, :] = (1 - lam) * w[a, :] + lam * z
    w_vec = w.flatten('C')
    Rlt = n * w_vec @ Omega_inv_2 @ w_vec.T
    RLT_II.append(Rlt)
print(RLT_II)
non_normal=[]
for i in range(Day_tra):
    if RLT_II[i] >= limit:
        non_normal.append(i)
print(non_normal)

T = len(P2_list)
plt.figure(figsize=(16,4))
plt.plot([i+1 for i in range(T)], np.log(RLT_II), 'b-',marker='*',label='$\ln$(QR-BR statistic)')
plt.plot([i+1 for i in range(T)], [np.log(limit) for i in range(T)], 'r-', label='$\ln$(control limit)')
plt.tick_params(axis='x', direction='inout')
plt.xlim(0,T+1)
#plt.xticks([i+1 for i in range(T)])
plt.xlabel('Day Number')
plt.legend()
plt.savefig('QR-BR_.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
#plt.title(j)
plt.show()
