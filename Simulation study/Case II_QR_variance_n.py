import numpy as np
from scipy.stats import multivariate_normal as M_n
import time
from IRLS import IRLS_List, generate_Omega

p = 5
Num = 1000 * 20
#Num = 1000

# 真实模型参数
eta_t = np.array([2,1,-3,-2,4]).reshape(-1,1) # regression coefficient
var_t = 1 # error variance

#训练数据,用于估计 Phase I 参数 Multiple
np.random.seed(2)
mean_x = np.zeros(p-1)
cov_x = np.array([[0.5,0.25,0.125,0.0625],[0.25,0.5,0.25,0.125],[0.125,0.25,0.5,0.25],[0.0625,0.125,0.25,0.5]])
print(cov_x)
x = np.random.multivariate_normal(mean=mean_x,cov=cov_x,size=Num).T
l_1 = np.array([1 for i in range(Num)])
X_0 = np.vstack((l_1, x)).T

# 随机误差
#Sceniro 1
err = np.random.normal(0, var_t, (Num, 1)) #normal errors

y_0 = X_0 @ eta_t + err


# IC 状态参数估计
tau_list = [0.1, 0.5, 0.9]
print(tau_list)
Cov = X_0.T @ X_0/Num
Omega = generate_Omega(p,tau_list,Cov)
print(Omega.shape)
Omega_inv = np.linalg.inv(Omega)

Beta_0 = IRLS_List(y_0,X_0,tau_list)
print(Beta_0)

#找控制线
nSimu = 10000 # 循环次数
nMaxRL = 3000 # ARL_0最大上限


def QR_Sum_sigma(limit, file_name,delta, lam=0.1, size=20):
    """

    :param limit: 控制线
    :param ld: EWMA系数
    :param N: 时间窗口宽度
    :param file_name:  保存文件名称
    :return:  无
    """
    nSimu = 10000  # 循环次数
    nMaxRL = 3000  # ARL_0最大上限

    Limit = limit  # 控制线
    n = size  # 样本个数
    print(delta)

    RL = []
    for i in range(nSimu):
        w = np.zeros((len(tau_list), p))

        k = 0
        Rlt = 0
        while (Rlt < Limit and k < nMaxRL):
            k += 1
            x = np.random.multivariate_normal(mean=mean_x,cov=cov_x,size=n).T
            l_1 = np.array([1 for i in range(n)])
            X = np.vstack((l_1, x)).T
            err_OC = (1 + delta) * np.random.normal(0, var_t, (n, 1))

            y = X @ eta_t + err_OC
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
        file.write("shift - %s \n" % (delta))
        file.write('ARL:\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}\n'.format(ARL, StdRL, Limit, lam))
    print('ARL:', ARL, StdRL, Limit, lam)


if __name__ == '__main__':
    file_name = './QR_variance_n'
    with open(file_name + '.txt', 'a') as file:
        file.write("normal N=20 *********************************************************************\n")

    #delta_list = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5] # [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
    delta_list = [-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3,0.4,0.5]
    size = 20
    lam_d = 0.1
    print(size)
    print(lam_d)
    limit = 1.6283 #2.436
    for i in delta_list:
        start_time = time.time()
        QR_Sum_sigma(limit=limit, file_name=file_name, delta= i, lam=lam_d, size=size)
        end_time = time.time()
        run_time = end_time - start_time
        with open(file_name + '.txt', 'a') as file:
            file.write("it  cost %s s for func running\n " % (round(run_time, 3)))