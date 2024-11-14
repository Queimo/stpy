from sklearn.base import BaseEstimator
import torch
import torch.optim as optim

from sklearn.preprocessing import StandardScaler

from stpy.kernels import KernelFunction
from stpy.continuous_processes.nystrom_fea import NystromFeatures
from stpy.helpers.helper import interval_torch
import torch
from torch.nn import Sequential, Linear
import torch.optim as optim
import numpy as np
import tqdm
from mtevi import EvidenceRegularizer, EvidentialnetMarginalLikelihood

from matplotlib import pyplot as plt

# def F(X):
#     x1, x2 = X[:, 0], X[:, 1]
#     x1 = 15 * x1 - 5
#     x2 = 15 * x2
#     a = 1
#     b = 5.1 / (4 * np.pi**2)
#     c = 5 / np.pi
#     r = 6
#     s = 10
#     t = 1 / (8 * np.pi)
#     return a * ((x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s) / 300


np.random.seed(10)
torch.manual_seed(10)

num_points = 20
a = [-3, -3]  # Lower bounds for each dimension
b = [3, 3]  # Upper bounds for each dimension
outlier_frac = 0.05

def F(X):
    x1, x2 = X[:, 0], X[:, 1]
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return a * ((x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s) / 300

# radial parabolic function
def F(X):
    x1, x2 = X[:, 0], X[:, 1]
    return x1**2 + x2**2

eps = lambda X: np.random.randn(X.shape[0])*0.1

def generate_data(F, eps=lambda X: 0, a=[-1, -1], b=[1, 1], num_points=100):
    # Generate X as a uniform distribution over the given range in each dimension
    X = np.random.uniform(low=a, high=b, size=(num_points, len(a)))
    Y = F(X) #+ eps(X)
    return X, Y

def add_outliers(X, Y, freq, a, b):
    n_outliers = int(freq * num_points)
    # Generate outliers in each dimension uniformly over the given range
    X_outliers = np.random.uniform(low=a, high=b, size=(n_outliers, len(a)))
    Y_outliers = F(X_outliers) + eps(X_outliers) + 5
    X_t = np.concatenate((X, X_outliers))
    Y_t = np.concatenate((Y, Y_outliers))
    return X_t, Y_t

# Generate data and add outliers
X, Y = generate_data(F, eps, a, b, num_points)
X_t, Y_t = add_outliers(X, Y, outlier_frac, a, b)



print(X_t.shape, Y_t.shape)

X_test, Y_test = generate_data(F, a=b, b=b, num_points=1000)

# kernel = KernelFunction(kernel_name="matern", gamma=0.5, nu = 3.5, d = 1) #TODO check why runtime is slow
# kernel = KernelFunction(kernel_name="squared_exponential", gamma=0.5, kappa=6.6,d = 1)
kernel = KernelFunction(kernel_name = "ard", d = 2, groups = [[0],[1]] )
# kernel = KernelFunction(kernel_name="squared_exponential", ard_gamma=[0.5, 0.5], kappa=[6.6, 6.6], d = 2)


def loss_fn(x,y,model):
    y_hat = model(x)
    y = y.double().unsqueeze(-1)
    # likelihood + prior
    return torch.mean((y_hat - y)**2) + 0.002*torch.sum(model.dense.weight**2)


def huber_loss(x,y,model, beta=1.):
    y_hat = model(x)
    y = y.double().unsqueeze(-1)
    return torch.nn.SmoothL1Loss(beta=beta)(y_hat, y)

objective = EvidentialnetMarginalLikelihood()
reg = EvidenceRegularizer(factor=0.0001)

def amini_loss(x,y,amini_model):
    output = amini_model(x)
    gamma, nu, alpha, beta = torch.chunk(output, 4, dim=-1)
    y = y.double().unsqueeze(-1)
    nll = (objective(gamma,nu,alpha,beta,y)).mean()
    reg_loss = (reg(gamma, nu, alpha, beta, y)).mean()
    return nll + reg_loss
    
def student_t_loss(x,y,student_t_model):
    output = student_t_model(x)
    mu, v, alpha = torch.chunk(output, 3, dim=-1)
    y = y.double().unsqueeze(-1)
    nll = torch.lgamma(v/2) + torch.log(torch.sqrt(torch.pi*v*alpha)) - torch.lgamma((v+1)/2) + ((v+1)/2)*torch.log((1 + (y-mu)**2/(v*alpha)))
    nll = nll.mean()
    return nll

class ShallowModel(Sequential):
    def __init__(self, input_dim, emb_dim ,output_dim):
        super(ShallowModel, self).__init__()
        
        self.embedding = NystromFeatures(kernel, m=emb_dim, approx="svd")
        self.emb_fit = False
        m = self.embedding.get_m()
        self.dense = Linear(m, output_dim, bias=False).double()
    
    def phi(self, x):
        if not self.emb_fit:
            # careful: only fit on first batch
            self.embedding.fit_gp(x,None, eps=0.1)
            self.emb_fit = True
            
        return self.embedding.embed(x)

    def forward(self, x):
        h = self.phi(x)
        y = self.dense(h)
        return y


class AminiModel(ShallowModel):
    def __init__(self, input_dim, emb_dim, output_dim):
        super(AminiModel, self).__init__(input_dim, emb_dim, output_dim * 4)
        
    def forward(self, x):
        h = self.phi(x)
        output = self.dense(h)
        mu, logv, logalpha, logbeta = torch.chunk(output, 4, dim=-1)
        v = torch.nn.functional.softplus(logv)
        alpha = torch.nn.functional.softplus(logalpha) + 1
        beta = torch.nn.functional.softplus(logbeta)
        return torch.cat([mu, v, alpha, beta], dim=-1)


class StudentTModel(ShallowModel):
    def __init__(self, input_dim, emb_dim, output_dim):
        super(StudentTModel, self).__init__(input_dim, emb_dim, output_dim * 3)
        
    def forward(self, x):
        h = self.phi(x)
        output = self.dense(h)
        mu, logv, logalpha = torch.chunk(output, 3, dim=-1)
        v = torch.nn.functional.softplus(logv) + 1
        alpha = torch.nn.functional.softplus(logalpha)
        return torch.cat([mu, v, alpha], dim=-1)


class PyTorchEstimator(BaseEstimator):
    def __init__(self, model_cls, criterion, optimizer_cls=optim.Adam, epochs=1000, lr=0.01):
        self.model_cls = model_cls
        self.model = None
        self.criterion = criterion
        self.optimizer_cls = optimizer_cls
        self.epochs = epochs
        self.lr = lr
    
    def fit(self, X, y):
        
        # Set model to training mode
        self.model = self.model_cls(input_dim=X.shape[1], emb_dim=X.shape[0]//5, output_dim=1).double()
        
        self.model.train()
        
        # Prepare data
        X_tensor = torch.tensor(X, dtype=torch.float64)
        y_tensor = torch.tensor(y, dtype=torch.float64)
        
        # Initialize optimizer
        optimizer = self.optimizer_cls(self.model.parameters(), lr=self.lr)
        
        #set_postfix loss print
        pbar = tqdm.tqdm(range(self.epochs))
        for i in pbar:
            optimizer.zero_grad()
            loss = self.criterion(X_tensor, y_tensor, self.model)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

    def predict(self, X):
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float64)
            outputs = self.model(X_tensor)
            if outputs.shape[-1] != 1:
                return outputs[:,0].numpy().squeeze()
        
        return outputs.numpy().squeeze()



# random shuffle X_t, Y_t 
perm = np.random.permutation(X_t.shape[0])
X_t = X_t[perm]
Y_t = Y_t[perm]

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor


# Wrap the model with our custom estimator
amini_estimator = PyTorchEstimator(model_cls=AminiModel, criterion=amini_loss)
student_t_estimator = PyTorchEstimator(model_cls=StudentTModel, criterion=student_t_loss)
gp_estimator = PyTorchEstimator(model_cls=ShallowModel, criterion=loss_fn)
huber_gp_estimator = PyTorchEstimator(model_cls=ShallowModel, criterion= huber_loss)
rf_estimator = RandomForestRegressor()

models_dict = {
    'Amini': amini_estimator,
    'Student T': student_t_estimator,
    'Normal GP': gp_estimator,
    'Huber GP': huber_gp_estimator,
    'Random Forest': rf_estimator
}

def run_model_cv(model, X, y, cv):
    # scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
    print(X.shape, y.shape)
    pred = cross_val_predict(model, X, y, cv=cv)
    print(pred.shape)
    res_dict = cross_validate(model, X, y, cv=cv, scoring='neg_mean_squared_error', return_estimator=True, return_train_score=True)
    res_dict['pred'] = pred
    return res_dict

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def test_model(model, X_t, Y_t, X_test, Y_test):
    model.fit(X_t, Y_t)
    pred = model.predict(X_test)
    mse = np.mean((pred - Y_test)**2)
    mae = MAE(Y_test, pred)
    return pred, X_test, mse, mae

def plot_cv_performance(pred, X, y, X_test, Y_test, pred_test, name):
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    axs = axs.flatten()
    F_real = F(X).squeeze()
    mse = (pred - F_real)**2
    # sorted_idx = np.argsort(X.flatten())
    # axs[0].plot(X[sorted_idx], y[sorted_idx], 'r.', label='Noisy Observations')
    # axs[0].plot(X[sorted_idx], pred[sorted_idx], 'b-', label='Predicted')
    # axs[0].plot(X[sorted_idx], F_real[sorted_idx], 'g-', label='True')
    
    axs[1].plot(F_real, pred, 'r.')
    axs[1].set_xlabel('True')
    axs[1].set_ylabel('Predicted')
    
    # axs[2].plot(X_test, Y_test, 'g-', label='True')
    # axs[2].plot(X_test, pred_test, 'b-', label='Predicted')
    # axs[2].plot(X[sorted_idx], y[sorted_idx], 'r.', label='Noisy Observations')
    
    
    axs[3].hist(mse, bins=20)
    axs[3].set_title('MSE')
    
    
    fig.suptitle(f'{name} MSE: {np.mean(mse):.4f}\n MAE: {MAE(F_real, pred):.4f}')
    return fig

from pathlib import Path
import datetime
data_dict = {}
figs = {}
todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
time_str = datetime.datetime.now().strftime('%H-%M-%S')
output_path = Path('output') / todays_date
output_path.mkdir(parents=True, exist_ok=True)
for model_name, model in models_dict.items():
    res_dict = run_model_cv(model, X_t, Y_t, cv=10)
    pred = res_dict['pred']
    scores = res_dict['test_score']
    
    #test model
    pred_test, X_test, mse, mae = test_model(model, X_t, Y_t, X_test, Y_test)
    
    print(f'{model_name}: {np.mean(scores)}')
    fig = plot_cv_performance(pred, X_t, Y_t, X_test, Y_test, pred_test, model_name)
    data_dict[model_name] = scores
    figs[model_name] = fig
    
    fig.savefig(output_path / f'{time_str}_{model_name}.png')

# pickle data_dict
import pickle
with open(output_path / f'{time_str}_data_dict.pkl', 'wb') as f:
    pickle.dump(data_dict, f)
   
# combine all images in figs dictonary into one image
from PIL import Image
images = [Image.open(output_path / f'{time_str}_{model_name}.png') for model_name in figs.keys()]
widths, heights = zip(*(i.size for i in images))
#make it square
max_width = max(widths)
total_height = sum(heights)
new_im = Image.new('RGB', (max_width, total_height))

y_offset = 0
for im in images:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1]
    
new_im.save(output_path / f'{time_str}_all_models.png')


# optimization of hyperparameters







