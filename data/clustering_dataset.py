import torch
from torch.utils.data import DataLoader, Dataset


class ClusteringDataset(Dataset):
    def __init__(self, x, y, target_probs):
        self.x = x
        self.y = y
        self.target_probs = target_probs

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.target_probs[index]

    @staticmethod
    def with_target_probs(old_x: torch.Tensor, old_y: torch.Tensor, target_probs: torch.Tensor, batch_size: int):
        return DataLoader(ClusteringDataset(old_x, old_y, target_probs), batch_size=batch_size, shuffle=True)
