import torch
from torch.utils.data.dataset import Dataset


class RandomIntDataset(Dataset):
    def __init__(self, length):
        self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return torch.randint(0, 32000, (1,))
