import os

import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.utils.tensorboard import SummaryWriter


def infer_kmeans(model, train_dataloader, test_dataloader, expected_num_clusters, device):
    x_train = torch.cat([y[0] for y in train_dataloader], 0).type(torch.float32).permute(0, 2, 1)
    y_train = torch.cat([y[1] for y in train_dataloader], 0).type(torch.float32)
    emb_train = model(x_train.to(device)).detach().type(torch.float32)

    x_test = torch.cat([y[0] for y in test_dataloader], 0).type(torch.float32).permute(0, 2, 1)
    y_test = torch.cat([y[1] for y in test_dataloader], 0).type(torch.float32)
    emb_test = model(x_test.to(device)).detach().type(torch.float32)

    kmeans = KMeans(n_clusters=expected_num_clusters, n_init=20)
    pred_train = torch.Tensor(kmeans.fit_predict(emb_train.cpu().numpy()))
    pred_test = torch.Tensor(kmeans.predict(emb_test.cpu().numpy()))

    print('===================')
    print('| KMeans train accuracy:', clustering_accuracy(y_train, pred_train).numpy(),
          '| KMeans test accuracy:', clustering_accuracy(y_test, pred_test).numpy())
    print('===================')
    return emb_train, emb_test, y_train, y_test, x_train, x_test


def clustering_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true, y_pred = y_true.int(), y_pred.int()
    total = y_pred.shape[0]
    sample_idx = torch.arange(total).to(y_true.device)
    indices = torch.stack([sample_idx, y_pred, y_true], 1).type(torch.int64)

    # add 1 when classes are ordered from 0
    pred_idx = int(torch.max(y_pred).item())
    true_idx = int(torch.max(y_true).item())
    if int(torch.min(y_pred).item()) == 0:
        pred_idx += 1
    if int(torch.min(y_true).item()) == 0:
        true_idx += 1

    idx = max(pred_idx, true_idx)
    data = torch.zeros([total, idx, idx], dtype=torch.float32)
    indices_by_columns = [indices[:, i] for i in range(indices.shape[-1])]
    data[indices_by_columns] = 1.
    data = data.sum(0)
    data = torch.max(data) - data
    row_ind, col_ind = linear_sum_assignment(data)
    gathered = data[row_ind, col_ind].float()
    return torch.sum(gathered) / total


def save_embeddings(log_dir, embeddings: torch.Tensor, labels: torch.Tensor, writer: SummaryWriter,
                    global_step: int = 0):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, f'metadata_{global_step}.tsv'), "w") as f:
        for l in labels:
            f.write(f'{l}\n')

    torch.save({"embedding": embeddings}, os.path.join(log_dir, f"embeddings_{global_step}"))
    writer.add_embedding(embeddings, labels, global_step=global_step)
