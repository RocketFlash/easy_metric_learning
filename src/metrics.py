import torch
import numpy as np
import pandas as pd

def GAP(pred, conf, true, return_x=False):
    '''
    Code from https://www.kaggle.com/davidthaler/gap-metric
    Compute Global Average Precision (aka micro AP), the metric for the
    Google Landmark Recognition competition. 
    This function takes predictions, labels and confidence scores as vectors.
    In both predictions and ground-truth, use None/np.nan for "no label".
    Args:
        pred: vector of integer-coded predictions
        conf: vector of probability or confidence scores for pred
        true: vector of integer-coded labels for ground truth
        return_x: also return the data frame used in the calculation
    Returns:
        GAP score
    '''
    x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})
    x.sort_values('conf', ascending=False, inplace=True, na_position='last')
    x['correct'] = (x.true == x.pred).astype(int)
    x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x['term'] = x.prec_k * x.correct
    gap = x.term.sum() / x.true.count()
    if return_x:
        return gap, x
    else:
        return gap

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