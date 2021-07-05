import torch
from torch.utils.data import DataLoader, Dataset

class vehicle_img(Dataset):
    """ A dataset containing data extracted from raw image. """
    
    def __init__(self, x, y, transform = None):
        self.X = x
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.transform(self.X[idx]), self.y[idx])
