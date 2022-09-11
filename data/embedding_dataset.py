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
        model.cpu()
        with torch.no_grad():
            x = [torch.FloatTensor(sample[0]) for sample in old_dataloader.dataset]
            x = torch.stack(x, 0).permute(0, 2, 1)
            embeddings = model(x).permute(0, 2, 1)
        model.to(device)
        return DataLoader(EmbeddingDataset(embeddings.cpu()), batch_size=old_dataloader.batch_size, shuffle=True)
