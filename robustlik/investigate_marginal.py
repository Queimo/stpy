import torch
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.kernels import KernelFunction
from stpy.helpers.helper import interval
import matplotlib.pyplot as plt
import numpy as np
import torch


# Parameters
N = 10
n = 256
d = 1
eps = 0.01
s = 1
gamma = .1


# Set random seed
torch.manual_seed(1)

# Generate random data
x = torch.rand(N, d).double() * 2 - 1
xtest = torch.from_numpy(interval(n, d, L_infinity_ball=1))

# True Gaussian process
GP_true = GaussianProcess(gamma=gamma, kernel_name="squared_exponential", d=d)
ytest = GP_true.sample(xtest)
GP_true.fit_gp(xtest, ytest)
y = GP_true.mean(x).clone()

# New data points
xnew = x[0, :].view(1, 1) + eps
ynew = y[0, 0].view(1, 1) + 1

# Combined data
x2 = torch.vstack([x, xnew])
y2 = torch.vstack([y, ynew])

# x2 = x
# y2 = y

#make 1 x 2 plot
fig, axs = plt.subplots(1,1)
loss = 'amini'

GPt = GaussianProcess(gamma=gamma, kernel_name="squared_exponential", d=d, loss=loss, lam=0.01)
GPt.fit_gp(x2, y2)
# GPt.optimize_params(type="bandwidth", restarts=6, verbose=True, optimizer='pytorch-minimize', scale=1., weight=1., bounds=[(0.01, 10.)])

params = GPt.kernel_object.get_param_refs()

gammas = np.linspace(0.01, .2, 50)


m = GPt._log_marginal_map(GPt.kernel_object, params, weight=1., return_components=True)

#replace all fields with np.arrays like gammas
for k, v in m.items():
    m[k] = np.zeros_like(gammas)
    
m["gamma"] = gammas


#parallelize
# with torch.no_grad():
#     for i, g in enumerate(gammas):
#         params["0"]["gamma"] = g
#         m_g = GPt._log_marginal_map(GPt.kernel_object, params, 1., return_components=True)
#         for k, v in m_g.items():
#             m[k][i] = v

from concurrent.futures import ThreadPoolExecutor

import time

#tick tock
start = time.time()

pool = ThreadPoolExecutor(max_workers=1)

def get_marginal(i, g):
    with torch.no_grad():
        params["0"]["gamma"] = g
        m_g = GPt._log_marginal_map(GPt.kernel_object, params, 1., return_components=True)
        
        for k, v in m_g.items():
            m[k][i] = v
 
for i, g in enumerate(gammas):
    pool.submit(get_marginal, i, g)
    
pool.shutdown(wait=True)  

print("Elapsed time: ", time.time() - start)

import pandas as pd
#plotly
import plotly.express as px

df = pd.DataFrame(m)

print(df)

fig = px.line(df, x="gamma", y=df.columns, title="Marginal likelihood components")

#date today
import datetime
today_time = datetime.datetime.now()
date_str = today_time.strftime("%Y-%m-%d-%H-%M-%S")

fig.write_html(f"mll_over_gamma_{date_str}_{loss}_N={N}_gamma={gamma}.html")

# mu_t, var_t, alea, epi = GPt.mean_std_sub(xtest)




















# #activate the first plot
# plt.sca(axs[0])

# plt.plot(xtest, mu_t, label="amini")
# plt.fill_between(xtest.squeeze(), mu_t.squeeze() - 2 * var_t.squeeze(), mu_t.squeeze() + 2 * var_t.squeeze(), alpha=0.2)
# # plt.fill_between(xtest.squeeze(), mu_t.squeeze() - 2 * alea.squeeze(), mu_t.squeeze() + 2 * alea.squeeze(), alpha=0.2, label="aleatoric")
# # plt.fill_between(xtest.squeeze(), mu_t.squeeze() - 2 * epi.squeeze(), mu_t.squeeze() + 2 * epi.squeeze(), alpha=0.2, label="epistemic")

# plt.plot(xtest, GP_true.mean(xtest), 'b--', label="truth", lw=1)
# plt.legend( bbox_to_anchor=(1., 1), loc='upper left')
# plt.plot(x, y, 'ro', ms=5, label="data")
# plt.plot(xnew, ynew, 'ko', ms=10, label="new data")


# GPt = GaussianProcess(gamma=gamma, kernel_name="squared_exponential", d=d, loss='amini', lam=0.000001)
# GPt.fit_gp(x2, y2)
# # GPt.optimize_params(type="bandwidth", restarts=5, verbose=True, optimizer='pytorch-minimize', scale=1., weight=1.)
# mu_t, var_t, alea, epi = GPt.mean_std_sub(xtest)
# plt.sca(axs[1])
# plt.plot(xtest, mu_t, 'r-', label="amini no reg")
# plt.fill_between(xtest.squeeze(), mu_t.squeeze() - 2 * var_t.squeeze(), mu_t.squeeze() + 2 * var_t.squeeze(), alpha=0.2, label="variance")
# plt.fill_between(xtest.squeeze(), mu_t.squeeze() - 2 * alea.squeeze(), mu_t.squeeze() + 2 * alea.squeeze(), alpha=0.2, label="aleatoric")
# plt.fill_between(xtest.squeeze(), mu_t.squeeze() - 2 * epi.squeeze(), mu_t.squeeze() + 2 * epi.squeeze(), alpha=0.2, label="epistemic")
# plt.plot(xtest, GP_true.mean(xtest), 'b--', label="truth", lw=1)
# plt.legend( bbox_to_anchor=(1., 1), loc='upper left')
# plt.plot(x, y, 'ro', ms=5, label="data")
# plt.plot(xnew, ynew, 'ko', ms=10, label="new data")



# plt.show()
