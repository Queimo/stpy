import torch
import pickle
import torch

class ODDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        data_dict = pickle.load(open(path, "rb"))
        df = data_dict["df"]
        self.X = torch.from_numpy(data_dict["X"]).to(dtype=torch.float64)
        self.meta = torch.from_numpy(df["OD"].values).to(dtype=torch.float64)
        self.y = torch.from_numpy(df["norm_TSNAK"].values).to(dtype=torch.float64)        
    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]
    def __len__(self):
        assert len(self.X) == len(self.y), "X and y have different lengths"
        return len(self.X)
    
if __name__ == "__main__":
    path = r".\data_exploration\data_set_dict.pkl"
    n_inp = 5
    ds = ODDataset(path, n_inp)
    print(ds[0])
    print(len(ds))


