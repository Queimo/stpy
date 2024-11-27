import torch
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.kernels import KernelFunction
from stpy.helpers.helper import interval

n = 30
d = 2
x = torch.rand(n,d).double()*2 - 1
xtest = torch.from_numpy(interval(50,d,L_infinity_ball=1))

GP = GaussianProcess(gamma=1121, kernel_name="ard", d=d)
y = GP.sample(x)
GP.fit_gp(x,y)
GP.visualize(xtest)

k = KernelFunction(kernel_name = "ard", d = d)
GP = GaussianProcess(kernel=k, loss="amini")
y = GP.sample(x)
GP.fit_gp(x,y)
GP.optimize_params(type="bandwidth", restarts=2, verbose = False, optimizer="pytorch-minimize")
print(GP.kernel.params)