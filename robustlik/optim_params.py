import torch
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.kernels import KernelFunction
from stpy.helpers.helper import interval
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


# Parameters
N = 10
n = 256
d = 1
eps = 0.01
s = 1
gamma = .2

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


lamdas = [.005]
# lamdas = np.logspace(-4, -2, 3)
# lamdas = np.linspace(1, 101, 10)
#increasing colors
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, len(lamdas))]
try:
    for i, lam in enumerate(lamdas):
        GP_student_corrupted = GaussianProcess(gamma=gamma, kernel_name="squared_exponential", d=d, loss='studentT', lam=lam)
        GP_student_corrupted.fit_gp(x2, y2)
        GP_student_corrupted.optimize_params(type="bandwidth", restarts=5, verbose=True, optimizer='pytorch-minimize', scale=1., weight=1.)
        mu_student_corrupted = GP_student_corrupted.mean(xtest)
        
        GP_huber_corrupted = GaussianProcess(gamma=gamma, kernel_name="squared_exponential", d=d, loss='huber', huber_delta=1.5, lam=.05)
        GP_huber_corrupted.fit_gp(x2, y2)
        # GP_huber_corrupted.optimize_params(type="bandwidth", restarts=5, verbose=True, optimizer='pytorch-minimize', scale=1., weight=1.)
        mu_huber_corrupted = GP_huber_corrupted.mean(xtest)
        
        if np.abs(mu_student_corrupted).max() > 10e6:
            plt.plot([], [], label=f'lam={lam:.4} --> nan', lw=1, color=colors[i])
        else:
            gamma = GP_student_corrupted.kernel_object.get_param_refs()["0"]['gamma'].item()
            print(GP_student_corrupted.kernel_object.get_param_refs()["0"])
            string = f'lam={lam:.4}\ngamma={gamma:.2}'
            plt.plot(xtest, mu_student_corrupted, label=string, lw=1, color=colors[i])
            
        plt.plot(xtest, mu_huber_corrupted, lw=1, color=colors[i], linestyle='--', label=f'huber')
except KeyboardInterrupt:
    pass

# Plotting
plt.plot(xtest, GP_true.mean(xtest), 'b--', label="truth", lw=1)
# plt.plot(xtest, mu_sqr_corrupted, 'r-', label="squared-corrupted", lw=1)
# # plt.plot(xtest, mu_sqr_uncorrupted, '--x', color="tab:brown", label='squared-uncorrupted', lw=3)
# plt.plot(xtest, mu_huber_corrupted, color="tab:green", label='huber-corrupted', lw=1)
# # plt.plot(xtest, mu_huber_uncorrupted, '--', color="tab:orange", label='huber-uncorrupted', lw=3)

# plt.plot(xtest, mu_student_corrupted, color="tab:blue", label='student-corrupted', lw=1)

# plt.plot(xtest, mu, '--', color="tab:purple", label='botorch ExactGP MLL', lw=1)

# Legend and display outside
plt.legend( bbox_to_anchor=(1., 1), loc='upper left')
# make figure larger to see the legend
plt.gcf().set_size_inches(10, 5)
# Plot data points
plt.plot(x, y, 'ro', ms=5, label="data")
plt.plot(xnew, ynew, 'ko', ms=10, label="new data")

plt.show()
