class MixCriterion:
    """Wrap a criterion so it can consume mixup/cutmix targets."""

    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(preds, targets1) + (1 - lam) * self.criterion(
            preds, targets2
        )
