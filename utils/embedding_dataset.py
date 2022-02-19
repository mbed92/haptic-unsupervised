import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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
