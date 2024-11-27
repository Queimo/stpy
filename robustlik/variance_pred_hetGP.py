
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
gamma = .5

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
fig, axs = plt.subplots(1, 2)
lambdas = [0.01]

for i, lam in enumerate(lambdas):
    GPt = GaussianProcess(gamma=gamma, kernel_name="squared_exponential", d=d, loss='hetGP', lam=lam)
    GPt.fit_gp(x2, y2)
    GPt.optimize_params(type="bandwidth", restarts=10, verbose=True, optimizer='pytorch-minimize', scale=1., weight=1.)
    mu_t, var_t  = GPt.mean_std_sub(xtest)

    #activate the first plot
    plt.sca(axs[0])

    plt.plot(xtest, mu_t, label="amini")
    plt.fill_between(xtest.squeeze(), mu_t.squeeze() - 2 * var_t.squeeze(), mu_t.squeeze() + 2 * var_t.squeeze(), alpha=0.2)

plt.plot(xtest, GP_true.mean(xtest), 'b--', label="truth", lw=1)
plt.legend( bbox_to_anchor=(1., 1), loc='upper left')
plt.plot(x, y, 'ro', ms=5, label="data")
plt.plot(xnew, ynew, 'ko', ms=10, label="new data")


GPt = GaussianProcess(gamma=gamma, kernel_name="squared_exponential", d=d, loss='hetGP', lam=lambdas[0])
GPt.fit_gp(x2, y2)
# GPt.optimize_params(type="bandwidth", restarts=5, verbose=True, optimizer='pytorch-minimize', scale=1., weight=1.)
mu_t, var_t = GPt.mean_std_sub(xtest)
plt.sca(axs[1])
plt.plot(xtest, mu_t, 'r-', label="amini no reg")
plt.fill_between(xtest.squeeze(), mu_t.squeeze() - 2 * var_t.squeeze(), mu_t.squeeze() + 2 * var_t.squeeze(), alpha=0.2, label="variance")
plt.plot(xtest, GP_true.mean(xtest), 'b--', label="truth", lw=1)
plt.legend( bbox_to_anchor=(1., 1), loc='upper left')
plt.plot(x, y, 'ro', ms=5, label="data")
plt.plot(xnew, ynew, 'ko', ms=10, label="new data")



plt.show()
