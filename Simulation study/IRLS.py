import numpy as np
from numpy.linalg import pinv
import timeit


def IRLS(tau,y,X,max_iter=100,p_tol=1e-6):
    beta = np.ones(X.shape[1]).reshape(-1,1)*10
    n_iter = 0
    xstar = X
    diff = 10
    while n_iter < max_iter and diff > p_tol:
        n_iter += 1
        beta0 = beta
        x_tx = np.dot(xstar.T, X)
        x_ty = np.dot(xstar.T, y)
        beta = np.dot(pinv(x_tx), x_ty)
        resid = y - np.dot(X, beta)
        mask = np.abs(resid) < .000001
        resid[mask] = ((resid[mask] >= 0) * 2 - 1) * .000001
        resid = np.where(resid < 0, tau * resid, (1 - tau) * resid)
        resid = np.abs(resid)
        xstar = X / resid
        diff = np.max(np.abs(beta - beta0))
    return beta

def IRLS_List(y,X,tau_list,max_iter=500,p_tol=1e-8):
    l=len(tau_list)
    beta_list = np.zeros(shape=(np.shape(X)[1], l))
    for i in range(l):
        beta_i = IRLS(tau_list[i], y, X,max_iter=max_iter,p_tol=p_tol)
        beta_list[:, i] = beta_i.squeeze()
    return beta_list

def small(a,b):
    if a<b:
        return a
    else:
        return b

def generate_Omega(p,tau_lis,Co):
    K = len(tau_lis)
    Omega = np.zeros((K * p, K * p))
    for i in range(K):
        for j in range(K):
            Omega_ij = (small(tau_lis[i], tau_lis[j]) - tau_lis[i] * tau_lis[j]) * Co
            row_start = i * p
            row_end = (i + 1) * p
            col_start = j * p
            col_end = (j + 1) * p
            Omega[row_start:row_end, col_start:col_end] = Omega_ij
    return Omega

"""
p = 2
n = 100
np.random.seed(2025)
#训练数据
l_1 = np.array([1 for i in range(n)]).reshape(-1, 1)
X = np.arange(1,n+1).reshape(-1,1)
X = np.hstack((l_1, X))
beta0 = np.array([1, 1]).reshape(-1, 1)
print(X[0:5])
print(X.shape)
err = np.random.normal(0, 0.5, (n, 1))
#err = np.random.laplace(0, 0.5, (n, 1))
#err = ald_rvs(0, 1, 0.3, (n,1))
y = X @ beta0 + err
start1 = timeit.default_timer()
tau_list = [0.1, 0.2,0.3, 0.4,0.5, 0.6,0.7,0.8, 0.9]
beta_list = IRLS_List(y,X,tau_list)
end1=timeit.default_timer()
print(beta_list)
print('IRLS### Running time: %s Seconds' % (end1 - start1))
"""