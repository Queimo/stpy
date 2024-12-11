import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL

import pickle

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import wandb

def parity_plot(y_true, y_pred):
    plt.scatter(y_true, y_pred)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    #square aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=1)
    plt.show()

# Load dataset
def load_dataset(path, prefilter=True):
    data_dict = pickle.load(open(path, "rb"))
    df = data_dict["df"]
    if prefilter:
        X = data_dict["X"][df["big_OD"]]
        y = df["norm_TSNAK"][df["big_OD"]].values
        groups = df["variant"][df["big_OD"]].values
        od = df["OD"][df["big_OD"]].values
    else:
        X = data_dict["X"]
        y = df["norm_TSNAK"].values
        groups = df["variant"].values
        od = df["OD"].values
        
    return X, y, groups, od

# this is for running the notebook in our testing framework
smoke_test = False

# path=r'.\data_exploration\data\ArM\data_set_dict.pkl'
path = './data_exploration/data/ArM/data_set_dict.pkl'
X, y, groups, od = load_dataset(path, prefilter=True)

splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 0)
train_idx, val_idx = next(splitter.split(X, y, groups))

# Use the indices to split the data
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
od_train, od_val = od[train_idx], od[val_idx]

#from validation set drop certain outliers which have OD < 0.018
X_val = X_val[od_val > 0.018]
y_val = y_val[od_val > 0.018]
od_val = od_val[od_val > 0.018]


# Split data into train and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Standardize y
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val = scaler_y.transform(y_val.reshape(-1, 1)).ravel()


if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = torch.tensor(X_train).cuda(), torch.tensor(y_train).cuda(), torch.tensor(X_val).cuda(), torch.tensor(y_val).cuda()
else:
    train_x, train_y, test_x, test_y = torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_val), torch.tensor(y_val)
    
#float32
train_x = train_x.float()
train_y = train_y.float()
test_x = test_x.float()
test_y = test_y.float()


from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)


class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


num_hidden_dims = 2 if smoke_test else 10


class DeepGP(DeepGP):
    def __init__(self, train_x_shape):
        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dims,
            mean_type='linear',
        )
        
        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        )
        
        super().__init__()
        
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()
    
    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output
    
    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))
        
        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)





# Initialize your DeepGP model
model = DeepGP(train_x.shape)
if torch.cuda.is_available():
    model = model.cuda()


# this is for running the notebook in our testing framework
num_epochs = 1 if smoke_test else 100
num_samples = 3 if smoke_test else 100


optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.01)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))

epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        with gpytorch.settings.num_likelihood_samples(num_samples):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

            minibatch_iter.set_postfix(loss=loss.item())
    
    epochs_iter.set_postfix(loss=loss.item())
    


import gpytorch
import math

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=1024)

model.eval()
predictive_means, predictive_variances, test_lls = model.predict(test_loader)

rmse = torch.mean(torch.pow(predictive_means.mean(0) - test_y, 2)).sqrt()
print(f"RMSE: {rmse.item()}, NLL: {-test_lls.mean().item()}")

from sklearn.metrics import r2_score

print(r2_score(test_y.cpu().numpy(), predictive_means.mean(0).cpu().numpy()))
parity_plot(test_y.cpu().numpy(), predictive_means.mean(0).cpu().numpy())