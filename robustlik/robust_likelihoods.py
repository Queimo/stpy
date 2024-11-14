from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.optim as optim

from stpy.kernels import KernelFunction
from stpy.continuous_processes.nystrom_fea import NystromFeatures
from stpy.helpers.helper import interval_torch, interval
import torch
from torch.nn import Sequential, Linear
import torch.optim as optim
import numpy as np
import tqdm
from mtevi import EvidenceRegularizer, EvidentialnetMarginalLikelihood


from matplotlib import pyplot as plt
from pathlib import Path
import datetime
import pickle
from PIL import Image

def generate_data(F, d, eps=lambda X: 0, a=-1, b=1, num_points=100):
    X = interval(num_points, d=2)
    Y = F(X) + eps(X)
    return X, Y

def generate_outliers(F, eps=lambda X: 0, a=-1, b=1, n_outliers=1, type='uniform'):
    X_outliers = np.random.uniform(a,b,n_outliers)
    y_max = 10
    y_min = -2
    if type == 'uniform':
        Y_outliers = np.random.uniform(y_min, y_max, n_outliers)
    elif type == 'bias+5':
        Y_outliers = F(X_outliers) + eps(X_outliers) + 5
    elif type == 'bias-5':
        Y_outliers = F(X_outliers) + eps(X_outliers) - 5
    elif type == 'random_bias':
        Y_outliers = F(X_outliers) + eps(X_outliers) + np.random.uniform(-5,5, n_outliers)    
    elif type == 'focused':
        X_outliers = np.random.normal((b+a)/2, (b-a)/40, n_outliers)
        Y_outliers = F(X_outliers) + eps(X_outliers) + 5
    
    X_outliers = np.expand_dims(X_outliers, 1)  
    Y_outliers = np.expand_dims(Y_outliers, 1)
    return X_outliers, Y_outliers

def mse_loss(x,y,model):
    y_hat = model(x)
    # likelihood + prior
    return torch.mean((y_hat - y)**2) + 0.002*torch.sum(model.dense.weight**2)

def huber_loss(x,y,model, beta=1.):
    y_hat = model(x)
    return torch.nn.SmoothL1Loss(beta=beta)(y_hat, y)

def amini_loss(x,y,amini_model):
    objective = EvidentialnetMarginalLikelihood()
    reg = EvidenceRegularizer(factor=0.0001)
    output = amini_model(x)
    gamma, nu, alpha, beta = torch.chunk(output, 4, dim=-1)
    y = y.double()
    nll = (objective(gamma,nu,alpha,beta,y)).mean()
    reg_loss = (reg(gamma, nu, alpha, beta, y)).mean()
    return nll + reg_loss
    
def student_t_loss(x,y,student_t_model):
    output = student_t_model(x)
    mu, v, alpha = torch.chunk(output, 3, dim=-1)
    y = y.double()
    nll = torch.lgamma(v/2) + torch.log(torch.sqrt(torch.pi*v*alpha)) - torch.lgamma((v+1)/2) + ((v+1)/2)*torch.log((1 + (y-mu)**2/(v*alpha)))
    nll = nll.mean()
    return nll

class ShallowModel(Sequential):
    def __init__(self, input_dim, emb_dim ,output_dim, kernel):
        super(ShallowModel, self).__init__()
        
        self.embedding = NystromFeatures(kernel, m=emb_dim, approx="nothing")
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
    def __init__(self, input_dim, emb_dim, output_dim, **kwargs):
        super(AminiModel, self).__init__(input_dim, emb_dim, output_dim * 4, **kwargs)
        
    def forward(self, x):
        h = self.phi(x)
        output = self.dense(h)
        mu, logv, logalpha, logbeta = torch.chunk(output, 4, dim=-1)
        v = torch.nn.functional.softplus(logv)
        alpha = torch.nn.functional.softplus(logalpha) + 1
        beta = torch.nn.functional.softplus(logbeta)
        return torch.cat([mu, v, alpha, beta], dim=-1)


class StudentTModel(ShallowModel):
    def __init__(self, input_dim, emb_dim, output_dim, **kwargs):
        super(StudentTModel, self).__init__(input_dim, emb_dim, output_dim * 3, **kwargs)
        
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
        self.kernel = KernelFunction(kernel_name="squared_exponential", gamma=0.5, kappa=6.6,d = 1)
    
    def fit(self, X, y):
        
        # Prepare data
        X_tensor = torch.tensor(X, dtype=torch.float64)
        y_tensor = torch.tensor(y, dtype=torch.float64)
        
        self.model = self.model_cls(kernel=self.kernel, input_dim=X.shape[1], emb_dim=X.shape[0], output_dim=1).double()
        self.model.train()
        
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


def run_model_cv(model, X, y, cv=10, n_jobs=-1):
    # scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
    print(X.shape, y.shape)
    
    pred = cross_val_predict(model, X, y, cv=cv, n_jobs=n_jobs)
    print(pred.shape)
    res_dict = cross_validate(model, X, y, cv=cv, 
                            scoring='neg_mean_squared_error', return_estimator=True,
                            return_train_score=True, n_jobs=n_jobs)
    res_dict['pred'] = pred
    return res_dict

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def test_model(model, X_t, Y_t, X_test, Y_test):
    model.fit(X_t, Y_t)
    pred = model.predict(X_test)
    mse = np.mean((pred - Y_test.squeeze())**2)
    mae = MAE(Y_test.squeeze(), pred)
    return pred, X_test, mse, mae

def plot_cv_performance(pred, X, y, X_test, Y_test, pred_test, name, test_mse, test_mae):
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    axs = axs.flatten()
    sorted_idx = np.argsort(X.flatten())
    F_real = F(X).squeeze()
    mse = (pred - F_real)**2
    axs[0].plot(X[sorted_idx], y[sorted_idx], 'r.', label='Noisy Observations')
    axs[0].plot(X[sorted_idx], pred[sorted_idx], 'b-', label='Predicted')
    axs[0].plot(X[sorted_idx], F_real[sorted_idx], 'g-', label='True')
    axs[0].set_title(f'{name} MSE: {np.mean(mse):.4f}\n MAE: {MAE(F_real, pred):.4f}')
    
    axs[1].plot(F_real, pred, 'r.')
    axs[1].set_xlabel('True')
    axs[1].set_ylabel('Predicted')
    #add symmetry line
    axs[1].plot([min(F_real), max(F_real)], [min(F_real), max(F_real)], 'k--')
    axs[1].set_aspect('equal', adjustable='box')
    
    axs[2].plot(X_test, Y_test, 'g-', label='True')
    axs[2].plot(X_test, pred_test, 'b-', label='Predicted')
    axs[2].plot(X[sorted_idx], y[sorted_idx], 'r.', label='Noisy Observations')
    axs[2].set_title(f'Test MSE: {test_mse:.4f}\n Test MAE: {test_mae:.4f}')
    
    
    axs[3].hist(mse, bins=20)
    axs[3].set_title('MSE')
    
    fig.suptitle(f'Inliers = {num_points}\n Outliers = {int(outlier_frac*num_points)}')
    return fig

if __name__ == '__main__':
    
    num_points = 30
    a = -3
    b = 10
    outlier_frac = 0.2
    
    d = 2
    
    outlier_types = ['uniform', 'bias+5', 'bias-5', 'random_bias', 'focused']
    outlier_types = ['focused']
    
    for outlier_type in outlier_types:
        np.random.seed(10)
        torch.manual_seed(10)

        # F = lambda X: np.sin(X*4)**3 + (X**2)/10
        # eps = lambda X: np.random.randn(*X.shape)*0.1 + 2*np.random.normal(scale=np.abs(7-np.abs(X))*0.05)

        # parabola nd
        def F(X):
            fac = 1/X.shape[-1]
            squares = np.sum(X**2, axis=1, keepdims=False)
            squares = np.expand_dims(squares, 1)
            return fac * squares
        eps = lambda X: 0
        
        X, Y = generate_data(F, d, eps, a, b, num_points)
        # X_outliers, Y_outliers = generate_outliers(F, eps, a, b, int(outlier_frac*num_points), type=outlier_type)

        # X_t = np.concatenate([X, X_outliers], axis=0)
        # Y_t = np.concatenate([Y, Y_outliers], axis=0)

        X_t, Y_t = X, Y
        
        # X_test, Y_test = generate_data(F, a=-6, b=13, num_points=500)
        X_test, Y_test = X, Y

        # random shuffle X_t, Y_t 
        perm = np.random.permutation(X_t.shape[0])
        X_t = X_t[perm]
        Y_t = Y_t[perm]

        # Wrap the model with our custom estimator
        amini_estimator = PyTorchEstimator(model_cls=AminiModel, criterion=amini_loss)
        student_t_estimator = PyTorchEstimator(model_cls=StudentTModel, criterion=student_t_loss)
        # gp_estimator = PyTorchEstimator(model_cls=ShallowModel, criterion=mse_loss)
        # huber_gp_estimator = PyTorchEstimator(model_cls=ShallowModel, criterion= huber_loss)
        # rf_estimator = RandomForestRegressor()

        models_dict = {
            'Amini': amini_estimator,
            'Student T': student_t_estimator,
            # 'Normal GP': gp_estimator,
            # 'Huber GP': huber_gp_estimator,
            # 'Random Forest': rf_estimator
        }
        
        data_dict = {}
        figs = {}
        todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
        time_str = datetime.datetime.now().strftime('%H-%M-%S')
        exp_name = f'{time_str}_num_points_{num_points}_outlier_frac_{outlier_frac}_type_{outlier_type}'
        output_path = Path('output') / todays_date
        output_path.mkdir(parents=True, exist_ok=True)

        for model_name, model in models_dict.items():
            res_dict = run_model_cv(model, X_t, Y_t)
            pred = res_dict['pred']
            scores = res_dict['test_score']

            # test model
            pred_test, X_test, test_mse, test_mae = test_model(model, X_t, Y_t, X_test, Y_test)

            print(f'{model_name}: {np.mean(scores)}')
            fig = plot_cv_performance(pred, X_t, Y_t, X_test, Y_test, pred_test, model_name, test_mse, test_mae)
            data_dict[model_name] = scores
            figs[model_name] = fig

            fig.savefig(output_path / f'{exp_name}_{model_name}.png')

        # pickle data_dict
        with open(output_path / f'{exp_name}.pkl', 'wb') as f:
            pickle.dump(data_dict, f)

        # combine all images in figs dictionary into one image
        images = [Image.open(output_path / f'{exp_name}_{model_name}.png') for model_name in figs.keys()]
        widths, heights = zip(*(i.size for i in images))

        # make it square
        max_height = max(heights)
        total_width = sum(widths)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_im.save(output_path / f'{exp_name}_allmodels.png')

        # delete image array
        for model_name in figs.keys():
            (output_path / f'{exp_name}_{model_name}.png').unlink()
