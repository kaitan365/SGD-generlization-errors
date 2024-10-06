# risk estimation of GD for (pseudo) Huber regression
# risk estimation of Proximal GD for (pseudo) Huber Lasso regression

import sys
import torch
import numpy as np
import cupy as cp
from cupy.linalg import norm
import cupyx.scipy.linalg as linalg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set_style('darkgrid')

print("Python version:\n", sys.version)
print("Torch version:", torch.__version__)
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("No CUDA GPUs are available, use CPU!")

dim_set = 1 # or 4
loss_set = 1 # or 2


if  dim_set == 1:
    n, p, T, lam = 10000, 5000, 500, 0 # no regularization
if  dim_set == 4:
    n, p, T, lam = 10000, 12000, 500, 0.002 # no regularization

# if loss_set == 0:
#     loss = 'least_square'
if loss_set == 1:
    loss = 'Huber'
if loss_set == 2:
    loss = 'pseudo_Huber'

print('loss, method, n, p, T, lam:\n', loss, 'GD', n, p, T, lam)


# learning rate
# eta = [1/cp.sqrt(t+1) for t in range(T)] # square-root schedule
# n = batch_size
eta = (1 + (p/n)**(0.5))**(-2) * cp.ones(T)

beta = cp.zeros(p)
s = int(p/20)
beta[:s] = 1
Sigma = cp.eye(p)#linalg.toeplitz(0.5 ** cp.arange(p))
signal = 10 
beta[:s] = beta[:s] * cp.sqrt(signal / (beta[:s].T @ Sigma[:s,:s] @ beta[:s]))
print('signal strength:', signal) 

def sqrt_mat_sym(M):
    # square root for symmetric matrix
    s, v = cp.linalg.eigh(M)
    sq_root = v * cp.sqrt(s) @ v.T
    return sq_root

Sigma_sq_root = sqrt_mat_sym(Sigma)

if loss == 'least_square':
    def psi(r):
        return r
    def psi_prime(r):
        return cp.ones_like(r)

if loss == 'Huber':
    def psi(r, delta=1):
        out = cp.where(cp.abs(r) <= delta, r, delta * cp.sign(r))
        return out
    def psi_prime(r, delta=1):
        out = cp.where(cp.abs(r) <= delta, 1, 0)
        return out

if loss == 'pseudo_Huber':
    def psi(r, delta=1):
        return r / cp.sqrt(1 + (r/delta)**2)
    def psi_prime(r, delta=1):
        return (1 + (r/delta)**2)**(-3/2)

def soft_threshold(x, threshold):
    return cp.sign(x) * cp.maximum(cp.abs(x) - threshold, 0)



def one_run(seed):
    # geenrate data
    cp.random.seed(seed)

    # use Hutchinson's trace approximation
    approx = True
    if approx:
        m0 = 2
        m = 2
        r0 = cp.random.choice([-1, 1], size=(p, m0))/cp.sqrt(m0)  # shape (p, m)
        r = cp.random.choice([-1, 1], size=(n, m))/cp.sqrt(m) # shape (n, m)
    else: # do not approximate
        m0 = p
        m = n
        r0 = cp.eye(p)
        r = cp.eye(n)
    r0TSigma = r0.T @ Sigma # required knowing Sigma

    X = cp.random.standard_normal((n,p)) #@ Sigma_sq_root
    eps = cp.random.standard_t(df=2, size=n)
    # eps = cp.random.laplace(loc=0, scale=2, size=n) # Laplace distribution

    y = X @ beta + eps
    # snr = norm(beta)**2/(norm(eps)**2/n)
    # print('snr:', snr)

    # run stochastic gradient descent
    # eta = n / norm(X, ord=2)**2 
    b = cp.zeros(p)
    B_hat = cp.zeros((p, T))

    for t in range(1, T):
        eta_t = eta[t-1]
        u = b+ eta_t/n * X.T @ psi(y - X @ b)  # GD
        b = soft_threshold(u, lam*eta_t)
        B_hat[:,t] = b

    H = Sigma_sq_root @ (B_hat - beta[:, cp.newaxis])
    R = y[:, cp.newaxis] - X @ B_hat
    F = psi(R)

    D = psi_prime(R)
    D_tr = cp.sum(D, 0)
    tD = (B_hat[:,1:] !=0).astype(int) # shape (p, T-1)
    tD = cp.hstack((tD, cp.zeros((p,1)))) # shape (p, T) # last column not used, just for the shape.
    assert cp.any(tD!=0) # avoid too large lambda such that all beta_hat are zero

    W = cp.zeros((T, T))
    tD_1 = (tD[:,0])[:, cp.newaxis]
    Q_2 = eta[0]/n * tD_1 * r0 # shape (p, m)
    Q_t_1 = Q_2
    W[1,0] = cp.trace(r0TSigma @ Q_2)


    A = cp.zeros((T,T))
    K = cp.zeros((T,T))
    XTr = X.T @ r

    R_2 = eta[0]/n * tD_1 * XTr # shape (p, m)
    R_t_1 = R_2
    A[1,0] = cp.trace(r.T * D[:,1] @ (X @ R_2))

    U_2 = eta[0]/n * tD_1 * X.T @ ((D[:,0])[:, cp.newaxis] * r)
    U_t_1 = U_2
    K[1,0] = cp.trace(r.T * D[:,1] @ (X @ U_2))

    for t in range(2, T):
        tD_t = (tD[:,t-1])[:, cp.newaxis]
        # Pt = - tD_t * (np.eye(p) - eta/n * X.T * D[:,t-1] @ X)
        # PQ = tD_t * (np.eye(p) - eta/n * X.T * D[:,t-1] @ X) @ Q_t_1
        PQ = tD_t * ( Q_t_1 - eta[t-1]/n * X.T @ ( (D[:,t-1])[:,cp.newaxis] * (X @ Q_t_1) ) )
        Q_t = cp.hstack((PQ, eta[t-1]/n * tD_t * r0))
        W[t,:t] = cp.einsum('jtj->t', (r0TSigma @ Q_t).reshape(m0, t, m0))
        Q_t_1 = Q_t

        PR = tD_t * (R_t_1 - eta[t-1]/n * X.T @ ( (D[:,t-1])[:,cp.newaxis] * (X @ R_t_1) ) )
        R_t = cp.hstack((PR, eta[t-1]/n * tD_t * XTr))
        temp = r.T * D[:,t] @ (X @ R_t) 
        A[t,:t] = cp.einsum('iti->t', temp.reshape(m, t, m))
        R_t_1 = R_t

        # Pt = tD_t * (np.eye(p) - eta/n * X.T * D[:,t-1] @ X)
        PU = tD_t * (U_t_1 - eta[t-1]/n * X.T @ ( (D[:,t-1])[:,cp.newaxis] * (X @ U_t_1) ) )
        U_t = cp.hstack((PU, eta[t-1]/n * tD_t * X.T @ ((D[:,t-1])[:, cp.newaxis] * r)))
        temp = r.T * D[:,t] @ (X @ U_t)
        K[t,:t] = cp.einsum('iti->t', temp.reshape(m, t, m))

        U_t_1 = U_t

    K = cp.diag(D_tr) - K
    W_hat = linalg.solve_triangular(K, A, lower=True)


    risk = norm(H, axis=0)**2 # + norm(eps)**2/n
    risk_est = norm(R + F @ W.T, axis=0)**2 /n - norm(eps)**2/n
    risk_est1 = norm(R + F @ W_hat.T, axis=0)**2 /n - norm(eps)**2/n

    dict = {
        'method': 'GD',
        'n': n,
        'p': p,
        'T': T,
        'lam': lam,
        'risk': risk.get() if isinstance(risk, cp.ndarray) else risk,
        'risk_estimate': risk_est.get() if isinstance(risk_est, cp.ndarray) else risk_est,
        'risk_estimate1': risk_est1.get() if isinstance(risk_est1, cp.ndarray) else risk_est1,
        'iteration': cp.arange(T).get()+1,
        'seed': seed
        }

    df = pd.DataFrame(dict)

    return df, cp.hstack((W, W_hat))

# try one experiment, get the time for one repetition 
import time
start_time = time.time()
a = one_run(0)
end_time = time.time()
executionime = end_time - start_time
print(f"one run: {executionime} seconds")

# try 100 repetitions
rep = 100
# from joblib import Parallel, delayed
# data_list = Parallel(n_jobs=-1, verbose=10)(
#     delayed(one_run)(seedid)
#     for seedid in range(rep)
#     )
data_list = []
for seedid in tqdm(range(rep)):
    result = one_run(seedid)
    data_list.append(result)

#####################################################
# save completed results
risk_list, W_list = zip(*data_list)

data_risk = pd.concat(risk_list, ignore_index=True)
data_risk.to_pickle(f'GD_data_risk_dim_{dim_set}_loss_{loss_set}.pkl' , compression='gzip')

data_W = cp.mean(cp.stack(W_list, axis=0), 0) # save the mean of W and W_hat
data_W = data_W.get()
np.save(f'GD_data_W_dim_{dim_set}_loss_{loss_set}.npy', data_W)
#####################################################


df_risk = pd.melt(data_risk, 
                  id_vars=['iteration', 'seed'],
                  value_vars=['risk', 'risk_estimate', 'risk_estimate1'],
                  var_name='Type')

#####################################################
# virtualization 
fs = 18 # fontsize
# risk plots
plt.figure(figsize=(6, 6))  # Set the figure size as appropriate
ax = sns.lineplot(data=df_risk,
             x='iteration', y='value', hue='Type',
             style="Type", 
             style_order=['risk','risk_estimate', 'risk_estimate1'],
             errorbar=("se", 2),
             legend='auto')
# plt.xscale('log')
plt.xlabel(r'Iteration number $t$', fontsize=fs)
plt.ylabel('Value', fontsize=fs)
handles, labels = ax.get_legend_handles_labels()
new_labels = [r'$\|\Sigma^{1/2}(\hat b^t - b^*)\|^2$', r'$\hat r_t - \|\epsilon\|^2/n$', r'$\tilde r_t-\|\epsilon\|^2/n$']
ax.legend(handles=handles, labels=new_labels, title='', fontsize=fs)
plt.tight_layout()
figname = f'GD_risk_dim_{dim_set}_loss_{loss_set}.pdf'    
plt.savefig(figname, bbox_inches='tight', dpi=300)
