import numpy as np
from scipy.stats import multivariate_normal as M_n
import time
from IRLS import IRLS_List, generate_Omega

p = 2
Num = 1000 * 20
#Num = 1000

# 真实模型参数
eta_t = np.array([3, 2]).reshape(-1, 1)  # regression coefficient
var_t = 1  # error variance

# 训练数据,用于估计 Phase I 参数
np.random.seed(2)
x = np.random.normal(loc=0, scale=0.5, size=Num)
l_1 = np.array([1 for i in range(Num)])
X_0 = np.vstack((l_1, x)).T

# 随机误差
# Sceniro 1
err = np.random.normal(0, var_t, (Num, 1))  # normal errors
print(np.mean(err))
print(np.std(err))
y_0 = X_0 @ eta_t + err

tau_list = [0.1, 0.5, 0.9]

# IC 状态参数估计
Cov = X_0.T @ X_0 / Num
Omega = generate_Omega(p, tau_list, Cov)
print(Omega.shape)
Omega_inv = np.linalg.inv(Omega)

Beta_0 = IRLS_List(y_0, X_0, tau_list)
print(Beta_0)

# 找控制线
nSimu = 10000  # 循环次数
nMaxRL = 3000  # ARL_0最大上限


def QR_Limit(limitL, limitR, file_name, lam=0.1, size=20):
    """
        :param limitL: 控制线下线
        :param limitR: 控制线上限
        :param file_name:  保存文件名
        :return: 无
        """
    start_time = time.time()
    LimitL = limitL
    LimitR = limitR
    n = size
    ARL = 220
    while abs(ARL - 200) >= 2:
        Limit = (LimitL + LimitR) / 2
        print(Limit)
        RL = []
        for i in range(nSimu):
            w = np.zeros((len(tau_list), p))

            k = 0
            Rlt = 0
            while (Rlt < Limit and k < nMaxRL):
                k += 1
                x = np.random.normal(loc=0, scale=0.5, size=n)
                l_1 = np.array([1 for i in range(n)])
                X = np.vstack((l_1, x)).T
                err = np.random.normal(0, var_t, (n, 1))
                y = X @ eta_t + err

                for a in range(len(tau_list)):
                    res = np.squeeze(y) - X @ Beta_0[:, a]
                    resid = np.where(res < 0, tau_list[a] - 1, tau_list[a])
                    z = np.sum(X.T * resid, axis=1) / n
                    w[a, :] = (1 - lam) * w[a, :] + lam * z
                w_vec = w.flatten('C')
                Rlt = n * w_vec @ Omega_inv @ w_vec.T

            RL.append(k)
        ARL = np.mean(RL)
        StdRL = np.std(RL) / np.sqrt(nSimu)
        with open(file_name + '.txt', 'a') as file:
            file.write('ARL:\t{:.4f},\t{:.4f},\t{:.4f}\n'.format(ARL, StdRL, Limit))
        print('ARL:', ARL, StdRL, Limit)
        if (ARL < 200):
            LimitL = Limit
        else:
            LimitR = Limit
    end_time = time.time()
    run_time = end_time - start_time
    print("it  cost %s s for func running " % (round(run_time, 3)))

    with open(file_name + '.txt', 'a') as file:
        file.write("it  cost %s s for func running\n " % (round(run_time, 3)))



file_name = "./QR_limit_n"
size = 20
lam_d = 0.1
print(size)
print(lam_d)
with open(file_name + '.txt', 'a') as file:
    file.write("n=20 chi\n")
QR_Limit(limitL=0.01, limitR=3.1738, file_name=file_name, lam=lam_d, size=size)
