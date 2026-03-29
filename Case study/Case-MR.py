import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sn
import statsmodels.api as sm

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

covariates = ['intercept','x1','x2','x3','x4']

print(DF1.shape)
print(DF2.shape)


def sigma_sum(X,Y,beta):
    sig = Y - X @ beta
    return np.sum(sig ** 2)


y_f1 = DF1['y'].to_numpy().reshape(-1,1)

x_f1 = DF1[covariates].to_numpy()

# IC 状态参数估计
eta_0 = np.linalg.inv(x_f1.T @ x_f1) @ x_f1.T @ y_f1
sigma_0 = sigma_sum(x_f1,y_f1,eta_0)/x_f1.shape[0]

print(eta_0)
print(sigma_0)

lam = 0.2

P1_list = DF1['Day_number'].unique()
outliers = [22, 23, 24, 25, 29, 30, 31, 33, 34,35]  # 1sigma

#
DF1 = DF1[~DF1['Day_number'].isin(outliers)]
#Bootstrap for control limit
D_l = DF1['Day_number'].unique()

# outliers
Rlt_I =[]


O =  x_f1.T @ x_f1/x_f1.shape[0]
c_ =  x_f1.T @ y_f1/x_f1.shape[0]
C = sigma_0
D = 78 * sigma_0

### residual
y_f1 = DF1['y'].to_numpy().reshape(-1,1)
x_f1 = DF1[covariates].to_numpy()
eta_0 = np.linalg.inv(x_f1.T @ x_f1) @ x_f1.T @ y_f1
res = y_f1 - x_f1 @ eta_0
print('***********************')
print(res.shape)
#plt.hist(res,bins=20)
plt.figure(figsize=(9,3))
g = sn.distplot(res,hist=True, kde=True,kde_kws={'linestyle':'--','linewidth':'1','color':'#c72e29', "label" : "KDE"},color='#098154',
           hist_kws={"histtype": "stepfilled" },axlabel='Residuals') #,axlabel='Xlabel',#设置x轴标题

plt.legend()
plt.show()

fig = sm.qqplot(res, line='45')
plt.show()



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
    eta_0 = np.linalg.inv(x_tra.T @ x_tra) @ x_tra.T @ y_tra
    sigma_0 = sigma_sum(x_tra,y_tra, eta_0) / x_tra.shape[0]

    O =  x_tra.T @ x_tra / x_tra.shape[0]
    c_ =  x_tra.T @ y_tra / x_tra.shape[0]
    C = sigma_0
    D = 78 * sigma_0

    # test
    for j in test_VN:
        data = test_da[test_da['Day_number'] == j]
        xi = data[covariates].to_numpy()
        yi = data['y'].to_numpy().reshape(-1,1)
        n = xi.shape[0]
        O = (1 - lam) * O + lam * xi.T @ xi/78
        c_ = (1 - lam) * c_ + lam * xi.T @ yi/78
        Eeta = np.linalg.inv(O) @ c_

        Esigma = sigma_sum(xi, yi, Eeta) / n
        C = (1 - lam) * C + lam * Esigma

        V = sigma_sum(xi, yi, eta_0)
        D = (1 - lam) * D + lam * V

        Rlt = n * np.log(sigma_0) - n * np.log(C) + D / sigma_0 - n

        Rlt_I.append(Rlt)
    RLT_Sum_list.extend(Rlt_I)

RLT_Sum_list = np.sort(RLT_Sum_list)[::-1]
limit = RLT_Sum_list[int(len(RLT_Sum_list) * 0.005)]
print(limit)

# without outliers

y_f1 = DF1['y'].to_numpy().reshape(-1,1)

x_f1 = DF1[covariates].to_numpy()

# IC 状态参数估计
eta_0 = np.linalg.inv(x_f1.T @ x_f1) @ x_f1.T @ y_f1
sigma_0 = sigma_sum(x_f1,y_f1,eta_0)/x_f1.shape[0]

print(eta_0)
print(sigma_0)



O = x_f1.T @ x_f1 / x_f1.shape[0]
c_ =  x_f1.T @ y_f1 / x_f1.shape[0]
C = sigma_0
D = 78 * sigma_0


P2_list = DF2['Day_number'].unique()
RLT_II =[]

for i in P2_list:
    data = DF2[DF2['Day_number'] == i]
    x2 = data[covariates].to_numpy()
    y2 = data['y'].to_numpy().reshape(-1,1)
    n = x2.shape[0]
    O = (1 - lam) * O + lam * x2.T @ x2/78
    c_ = (1 - lam) * c_ + lam * x2.T @ y2/78
    Eeta = np.linalg.inv(O) @ c_

    Esigma = sigma_sum(x2, y2, Eeta) / n
    C = (1 - lam) * C + lam * Esigma

    V = sigma_sum(x2, y2, eta_0)
    D = (1 - lam) * D + lam * V

    Rlt = n * np.log(sigma_0) - n * np.log(C) + D / sigma_0 - n

    RLT_II.append(Rlt)
print(RLT_II)

T = len(P2_list)
plt.figure(figsize=(16,4))
plt.plot([i+1 for i in range(T)], np.log(RLT_II), 'b-',marker='.',label='$\ln$(MR statistic)')
plt.plot([i+1 for i in range(T)], [np.log(limit) for i in range(T)], 'r-', label='$\ln$(control limit)')
plt.tick_params(axis='x', direction='inout')
plt.xlim(0,T+1)
plt.xlabel('Day Number')
plt.legend()
#plt.savefig('MR.png', dpi=200 ,bbox_inches='tight')
plt.show()
