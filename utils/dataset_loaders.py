import torch
import torch.nn as nn
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


class EmbeddingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], {}  # keep compatible with HapticDataset, etc.

    @staticmethod
    def gather_embeddings(model: nn.Module, old_dataloader: DataLoader, device: int):
        model.train(False)
        x = torch.cat([y[0] for y in old_dataloader], 0).to(device)
        x = x.type(torch.float32).permute(0, 2, 1)
        embeddings = model(x).permute(0, 2, 1)
        return DataLoader(EmbeddingDataset(embeddings.detach()), batch_size=old_dataloader.batch_size, shuffle=True)
