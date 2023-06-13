import torch

def accuracy(output, labels):
    if isinstance(labels, (tuple, list)):
        targets1, targets2, lam = labels
        _ , preds = torch.max(output, dim=1)
        correct1 = preds.eq(targets1).sum().item()
        correct2 = preds.eq(targets2).sum().item()
        return (lam*correct1+(1-lam)*correct2)/preds.size(0)
    else:
        _,pred = torch.max(output, dim=1)
        return torch.sum(pred==labels).item()/labels.size(0)
    

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