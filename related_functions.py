import numpy as np
from scipy import stats
from scipy.linalg import norm
from scipy.linalg import inv, sqrtm
from scipy.optimize import minimize
from IRLS import IRLS
import warnings
import os

def wilcoxon_rank1(X, y, intercept, max_iter=100, tol=1e-6):
    """
    IRLS算法求解Wilcoxon秩回归（已知截距）

    参数:
    X: 特征矩阵 (n_samples, n_features)
    y: 响应变量 (n_samples,)
    intercept: 已知截距
    max_iter: 最大迭代次数
    tol: 收敛容忍度

    返回:
    斜率系数向量
    """
    n, p = X.shape

    # 初始OLS估计
    y_adj = y - intercept
    beta = np.linalg.lstsq(X, y_adj, rcond=None)[0]

    for _ in range(max_iter):
        # 计算残差
        residuals = y_adj - X @ beta

        # 计算秩得分
        ranks = stats.rankdata(residuals)
        scores = np.sqrt(12) * (ranks / (n + 1) - 0.5)
        sorted_scores = scores[np.argsort(np.argsort(residuals))]

        # 计算权重
        weights = np.abs(sorted_scores)
        weights = np.maximum(weights, 1e-8)

        # 加权最小二乘更新
        W_sqrt = np.sqrt(weights)
        X_weighted = X * W_sqrt[:, np.newaxis]
        y_weighted = y_adj * W_sqrt

        beta_new = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)[0]

        # 检查收敛
        if norm(beta_new - beta) < tol:
            break

        beta = beta_new

    return beta


def spacial_sign_R(matrix_x):
    os.environ['R_HOME'] = 'C:\Program Files\R\R-4.0.2'
    #os.environ['R_USER'] = 'D:\Program Files\Python\Python37\Lib\site-packages\rpy2'
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr

    rpy2.robjects.numpy2ri.activate()
    ICSNP = importr('ICSNP')

    data_set = np.array(matrix_x)

    nr,nc = data_set.shape
    data_set_r = robjects.r.matrix(data_set,nrow=nr, ncol=nc)
    result = tuple(ICSNP.HR_Mest(data_set_r))
    median_1 = result[0]
    scatter = result[1]
    scatter_inv = np.linalg.inv(scatter)
    A_0 = np.linalg.cholesky(scatter_inv)
    A_0 = A_0 / A_0[0,0]
    return median_1,A_0.T


def wilcoxon_rank(X, y, max_iter=100, epsilon=1e-5, verbose=False):
    """
    计算Wilcoxon秩估计(WRE) - 无截距版本


    """

    X = np.asarray(X)
    y = np.asarray(y)
    n, p = X.shape

    if n <= p:
        raise ValueError("样本量必须大于特征数")

    # 步骤1: 使用最小二乘估计作为初始值 (无截距)
    beta_current = np.linalg.lstsq(X, y, rcond=None)[0]

    if verbose:
        print(f"初始OLS估计: {beta_current}")

    iterations = 0
    converged = False

    for iter_num in range(max_iter):
        iterations = iter_num + 1

        # 计算当前残差
        residuals = y - X @ beta_current

        # 计算残差中位数
        m_beta = np.median(residuals)

        # 计算残差的秩
        ranks = stats.rankdata(residuals)

        # 计算T_i(beta) = R(y_i - x_i^T beta)/(n+1) - 1/2
        T = ranks / (n + 1) - 0.5

        # 计算权重 w_i(beta)
        w = np.zeros(n)
        for i in range(n):
            if residuals[i] != m_beta:  # 避免除以零
                w[i] = T[i] / (residuals[i] - m_beta)
            else:
                w[i] = 0

        # 检查权重矩阵是否可逆
        W = np.diag(w)
        try:
            # 计算更新项: (X^T W X)^(-1) X^T W (r - m)
            XTWX = X.T @ W @ X
            XTWX_inv = np.linalg.inv(XTWX)
            update_term = XTWX_inv @ X.T @ W @ (residuals - m_beta)
        except np.linalg.LinAlgError:
            #warnings.warn("矩阵不可逆，使用伪逆")
            XTWX_inv = np.linalg.pinv(XTWX)
            update_term = XTWX_inv @ X.T @ W @ (residuals - m_beta)

        # 更新beta
        beta_new = beta_current + update_term

        # 计算目标函数值(分散函数)
        W_n_current = np.sum(T * (residuals - m_beta))

        # 计算新残差和目标函数值用于收敛检查
        residuals_new = y - X @ beta_new
        m_beta_new = np.median(residuals_new)
        ranks_new = stats.rankdata(residuals_new)
        T_new = ranks_new / (n + 1) - 0.5
        W_n_new = np.sum(T_new * (residuals_new - m_beta_new))

        # 检查收敛
        if abs(W_n_new - W_n_current) <= epsilon:
            converged = True
            if verbose:
                print(f"在 {iterations} 次迭代后收敛")
                print(f"最终目标函数值: {W_n_new:.6f}")
            break

        beta_current = beta_new

        if verbose and iter_num % 100 == 0:
            print(f"迭代 {iterations}: W_n = {W_n_new:.6f}")

    if not converged:
        pass

    return beta_current


def estimate_sq(X,y, q, bandwidth_method='h3', alpha=0.05):
    """
    估计稀疏函数 s(q) = 1 / f(F^{-1}(q))

    Parameters:
    -----------
    samples : list, 历史样本
    q : float, 分位点
    bandwidth_method : str, 带宽选择方法 ('h1', 'h2', 'h3')
    alpha : float, 显著性水平 (用于h2, h3)

    Returns:
    --------
    s_q : float, 稀疏函数估计值
    """
    # 使用差分商方法估计稀疏函数
    n = X.shape[0] # 样本大小

    # 选择带宽
    h_n = select_bandwidth(n, q, bandwidth_method, alpha)


    x_bar = np.mean(X, axis=0)
    beta_q_plus = IRLS(min(q + h_n, 0.99), y, X)

    beta_q_minus = IRLS(max(q - h_n, 0.01),y, X)
    s_est = np.squeeze(beta_q_plus - beta_q_minus) @ x_bar / (2 * h_n)

    return s_est


def select_bandwidth(n, q, method='h3', alpha=0.05):
    """
    选择带宽 h_n

    Parameters:
    -----------
    n : int, 样本大小
    q : float, 分位点
    method : str, 带宽选择方法
    alpha : float, 显著性水平

    Returns:
    --------
    h_n : float, 带宽
    """
    if method == 'h1':
        # Bofinger (1975) 带宽
        from scipy.stats import norm
        z_q = norm.ppf(q)
        phi_z = norm.pdf(z_q)
        numerator = 4.5 * phi_z ** 4
        denominator = (2 * z_q ** 2 + 1) ** 2
        return n ** (-1 / 5) * (numerator / denominator) ** (1 / 5)

    elif method == 'h2':
        # Hall and Sheather (1988) 带宽
        from scipy.stats import norm
        z_q = norm.ppf(q)
        z_alpha = norm.ppf(1 - alpha / 2)
        phi_z = norm.pdf(z_q)
        numerator = 1.5 * phi_z ** 2
        denominator = 2 * z_q ** 2 + 1
        return n ** (-1 / 3) * z_alpha ** (2 / 3) * (numerator / denominator) ** (1 / 3)

    elif method == 'h3':
        # Chamberlain (1994) 带宽
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha / 2)
        return z_alpha * np.sqrt(q * (1 - q) / n)

    else:
        raise ValueError("不支持的带宽选择方法")


def build_omega(taus, s_values):
    """
    构建 Omega 矩阵

    Parameters:
    -----------
    taus : list, 分位点列表
    s_values : list, 对应的稀疏函数值列表

    Returns:
    --------
    Omega : array, Omega矩阵
    """
    k = len(taus)
    Omega = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            tau_i, tau_j = taus[i], taus[j]
            s_i, s_j = s_values[i], s_values[j]
            Omega[i, j] = (min(tau_i, tau_j) - tau_i * tau_j) * s_i * s_j

    return Omega


def build_R(p):
    """
    构建 R 矩阵
    """
    # 创建一个10×15的零矩阵作为基础
    R = np.zeros((2*p, 3*p), dtype=int)

    # 1. 左上角5×5单位矩阵（0-4行，0-4列）
    R[0:p, 0:p] = np.eye(p, dtype=int)

    # 2. 5-10行与5-10列的5×5单位矩阵（5-9行，5-9列）
    R[p:2*p, p:2*p] = np.eye(p, dtype=int)

    # 3. 5-10行与10-15列的单位矩阵乘-1（5-9行，10-14列）
    R[p:2*p, 2*p:3*p] = -np.eye(p, dtype=int)
    return R