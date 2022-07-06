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