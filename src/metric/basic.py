import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def recall_at_k(y_true, y_pred, k=1):
    return np.equal(y_pred[:, :k], y_true[:, None]).any(axis=1).mean()


def mean_average_precision_at_r(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    average_precisions = []

    for index, label in enumerate(y_true):
        r = int(np.sum(y_true == label) - 1)
        if r <= 0:
            continue

        ranked_predictions = y_pred[index, :r]
        relevant = ranked_predictions == label
        if relevant.size == 0:
            average_precisions.append(0.0)
            continue

        precision_at_k = np.cumsum(relevant) / (np.arange(relevant.size) + 1)
        average_precisions.append(float(np.sum(precision_at_k * relevant) / r))

    if not average_precisions:
        return 0.0
    return float(np.mean(average_precisions))


def nearest_label_nmi(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 0]
    return float(normalized_mutual_info_score(y_true, y_pred))


def accuracy(output, labels):
    if isinstance(labels, (tuple, list)):
        targets1, targets2, lam = labels
        _, preds = torch.max(output, dim=1)
        correct1 = preds.eq(targets1).sum().item()
        correct2 = preds.eq(targets2).sum().item()
        return (lam * correct1 + (1 - lam) * correct2) / preds.size(0)
    else:
        _, pred = torch.max(output, dim=1)
        return torch.sum(pred == labels).item() / labels.size(0)


def f_score(output, label, threshold=0.5, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)
