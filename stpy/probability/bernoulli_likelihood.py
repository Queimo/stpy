import cvxpy as cp
import numpy as np
import torch
from typing import Union, Dict, List
from stpy.probability.likelihood import Likelihood
from stpy.probability.gaussian_likelihood import GaussianLikelihood
import scipy

class BernoulliLikelihoodCanonical(GaussianLikelihood):

    def __init__(self):
        super().__init__()

    def evaluate_datapoint(self, theta, d, mask = None):
        if mask is None:
            mask = 1.
        x, y = d
        r = -y*(x@theta) + torch.log(1+torch.exp(x@theta))
        r = r * mask
        return r

    def link(self, s):
        return 1./(1.+ torch.exp(-s))

    def scale(self, mask = None):
        return 1.

    def get_objective_cvxpy(self, mask = None):
        if mask is None:
            def likelihood(theta):
                return -self.y.T@(self.x @ theta) + cp.sum(cp.logistic(self.x @ theta))
        else:
            def likelihood(theta):
                if torch.sum(mask.double())>1e-8:
                    return -(mask*self.y)@(self.x @ theta) + mask @ cp.logistic(self.x @ theta)
                else:
                    return cp.sum(theta*0)
        return likelihood

    def lipschitz_constant(self, b):
        return np.exp(b)

    def get_confidence_set_cvxpy(self,
                                 theta: cp.Variable,
                                 type: Union[str, None] = None,
                                 params: Dict = {},
                                 delta: float = 0.1):
        if self.fitted == True:
            return self.set_fn(theta)

        theta_fit = params['estimate']
        H = params['regularizer_hessian']
        lam = torch.max(torch.linalg.eigvalsh(H))
        B = params['bound']
        d_eff = params['d_eff']

        if type in ['faubry']:
            D = torch.diag(1./(self.x @ theta_fit).view(-1))
            V = self.x.T @ D @ self.x + H

            beta = np.sqrt(lam*B) / 2. + 2. / np.sqrt(lam*B) * (torch.logdet(V) - torch.logdet(H)) + 2 / np.sqrt(
                lam*B) * np.log(1 / delta) * d_eff

            L = torch.from_numpy(scipy.linalg.sqrtm(V.numpy()))
            self.set_fn = lambda theta: [cp.sum_squares(L @ (theta - theta_fit)) <= beta]
            set = self.set_fn(theta)

        elif type in ['laplace']:
            sigma = 1./4.
            V = self.x.T @ self.x / sigma**2 + H
            L = torch.from_numpy(scipy.linalg.sqrtm(V.numpy()))
            beta = 2. * self.lipschitz_constant(B)
            self.set_fn = lambda theta: [cp.sum_squares(L @ (theta - theta_fit)) <= beta]
            set = self.set_fn(theta)

        elif type in ["adaptive-AB"]:
            sigma = 1./4.
            V = self.x.T @ self.x / sigma**2 + H
            L = torch.from_numpy(scipy.linalg.sqrtm(V.numpy()))
            beta = 2 * np.log(1. / delta) + (torch.logdet(V + H) - torch.logdet(H)) + lam * B
            self.set_fn = lambda theta:  [cp.sum_squares(L@(theta - theta_fit)) <= beta]
            set = self.set_fn(theta)

        elif type == "LR":
            beta = self.confidence_parameter(delta, params, type=type)
            set = self.lr_confidence_set_cvxpy(theta, beta, params)

        else:
            raise NotImplementedError("The desired confidence set type is not supported.")

        self.set = set
        self.fitted = True

        return set

    def information_matrix(self):
        V = self.x.T@self.x/self.sigma
        return V

    def confidence_parameter(self, delta, params, type = None):
        H = params['regularizer_hessian']
        lam = torch.max(torch.linalg.eigvalsh(H))
        B = params['bound']
        d_eff = params['d_eff']

        if type is None or type == "none" or type == "laplace":
            # this is a common heuristic
            beta =  2.0

        elif type == "adaptive-AB":
            sigma = 1./4.
            V = self.x.T @ self.x / sigma ** 2 + H
            beta = 2 * np.log(1. / delta) + (torch.logdet(V + H) - torch.logdet(H)) + lam * B

        elif type == "LR":
            # this is based on sequential LR test
            beta = self.confidence_parameter_likelihood_ratio(delta, params)

        elif type == "Faubry":
            H = params['regularizer_hessian']
            lam = H[0., 0]
            theta_fit = params['estimate']
            D = torch.diag(1./(self.x @ theta_fit).view(-1))
            V = self.x.T @ D @ self.x + H
            beta = np.sqrt(lam)/2. + 2./np.sqrt(lam)*(torch.logdet(V) - torch.logdet(H)) + 2/np.sqrt(lam)* np.log(1/delta)*d_eff
        else:
            raise NotImplementedError("Not implemented")
        return beta

    def get_objective_torch(self):
        raise NotImplementedError("Implement me please.")
