import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from related_functions import wilcoxon_rank,spacial_sign_R
import random

def sigma_sum(X,Y,beta,a):
    sig = Y - X @ beta - a
    return np.sum(sig ** 2)

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

lam = 0.2
p=5
covariates = ['x1','x2','x3','x4']

y_f1 = DF1['y'].to_numpy().reshape(-1,1)

x_f1 = DF1[covariates].to_numpy()


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

    m = int(x_tra.shape[0] / 78)
    Z_matrix = np.zeros((m, p + 1))
    for t in range(m):
        y_0_ = y_tra[t * 78:(t + 1) * 78, :]
        x_ = x_tra[t * 78:(t + 1) * 78, :]

        a_0 = np.mean(y_0_)

        b_1 = wilcoxon_rank(x_, np.squeeze(y_0_))

        sigma_0 = sigma_sum(x_, y_0_, b_1.reshape(-1, 1), a_0) / (78 - p)
        z = np.concatenate([[a_0], np.array(b_1), [sigma_0]])
        Z_matrix[t, :] = z


    med, S = spacial_sign_R(Z_matrix)

    # test
    #w = np.zeros(p + 1)
    for j in test_VN:
        data = test_da[test_da['Day_number'] == j]
        xi = data[covariates].to_numpy()
        yi = data['y'].to_numpy().reshape(-1,1)
        n = xi.shape[0]

        a_c = np.mean(yi)

        b_1_c = wilcoxon_rank(xi, np.squeeze(yi))

        sigma_c = sigma_sum(xi, yi, b_1_c.reshape(-1, 1), a_c) / (n - p)

        z = np.concatenate([[a_c], np.array(b_1_c), [sigma_c]])
        z = (z - med) @ S.T
        z = z / np.linalg.norm(z)

        w = (1 - lam) * w + lam * z

        Rlt = (2 - lam) / lam * (p + 1) * w @ w.T
        Rlt_I.append(Rlt)
    RLT_Sum_list.extend(Rlt_I)

RLT_Sum_list = np.sort(RLT_Sum_list)[::-1]
limit = RLT_Sum_list[int(len(RLT_Sum_list) * 0.005)]
limit = 33.8
print(limit)

# phase II

y_f1 = DF1['y'].to_numpy().reshape(-1,1)

x_f1 = DF1[covariates].to_numpy()

m = int(x_f1.shape[0]/78)

Z_matrix = np.zeros((m,p+1))
for t in range(m):
    y_0 = y_f1[t*78:(t+1)*78,:]
    x = x_f1[t*78:(t+1)*78,:]

    a_0 = np.mean(y_0)

    b_1 = wilcoxon_rank(x,np.squeeze(y_0))

    sigma_0 = sigma_sum(x,y_0,b_1.reshape(-1,1),a_0)/(78-p)
    z = np.concatenate([[a_0], np.array(b_1), [sigma_0]])
    Z_matrix[t, :] = z
print(Z_matrix.shape)

med, S = spacial_sign_R(Z_matrix)

P2_list = DF2['Day_number'].unique()
RLT_II =[]


w = np.zeros(p + 1)

for i in P2_list:
    data = DF2[DF2['Day_number'] == i]
    x2 = data[covariates].to_numpy()
    y2 = data['y'].to_numpy().reshape(-1,1)
    n = x2.shape[0]

    a_c = np.mean(y2)

    b_1_c = wilcoxon_rank(x2, np.squeeze(y2))

    sigma_c = sigma_sum(x2, y2, b_1_c.reshape(-1, 1), a_c) / (n - p)

    z = np.concatenate([[a_c], np.array(b_1_c), [sigma_c]])
    z = (z - med) @ S.T
    z = z / np.linalg.norm(z)

    w = (1 - lam) * w + lam * z

    Rlt = (2 - lam) / lam * (p + 1) * w @ w.T
    RLT_II.append(Rlt)
print(RLT_II)
non_normal=[]
for i in range(Day_tra):
    if RLT_II[i] >= limit:
        non_normal.append(i)
print(non_normal)

T = len(P2_list)
plt.figure(figsize=(16,4))
plt.plot([i+1 for i in range(T)], np.log(RLT_II), 'b-',marker='.',label='$\ln$(RR statistic)')
plt.plot([i+1 for i in range(T)], [np.log(limit) for i in range(T)], 'r-', label='$\ln$(control limit)')
plt.tick_params(axis='x', direction='inout')
plt.xlim(0,T+1)
#plt.xticks([i+1 for i in range(T)])
plt.xlabel('Day Number')
plt.legend()
#plt.savefig('Rank_.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
#plt.title(j)
plt.show()