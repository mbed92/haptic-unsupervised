import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import utils


class EmbeddingDataset(Dataset):
    @staticmethod
    def gather_embeddings(model: nn.Module, old_dataloader: DataLoader, device: int, input_data_shape: list):
        model.train(False)

        dataset = list()
        for data in old_dataloader:
            batch_data, _ = utils.dataset.load_samples_to_device(data, device)
            batch_data = batch_data.reshape([-1, *input_data_shape])
            batch_embeddings = model(batch_data)
            for embedding in batch_embeddings:
                dataset.append(embedding)
        return DataLoader(dataset, batch_size=old_dataloader.batch_size, shuffle=True)
