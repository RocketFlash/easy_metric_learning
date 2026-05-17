import torch
import numpy as np


def l2_norm(input_x, axis=1):
    norm = torch.norm(input_x, 2, axis, True)
    output = torch.div(input_x, norm)
    return output


def is_mixed_label(label):
    return isinstance(label, (list, tuple))


def get_primary_label(label):
    if is_mixed_label(label):
        return label[0]
    return label


def build_one_hot(label, num_classes, device, dtype=torch.float32):
    """Build one-hot or mixed-label weights for margin heads."""
    if is_mixed_label(label):
        label1, label2, lam = label
        one_hot = torch.zeros(label1.size(0), num_classes, device=device, dtype=dtype)
        lam1 = torch.full((label1.size(0), 1), float(lam), device=device, dtype=dtype)
        lam2 = torch.full(
            (label2.size(0), 1), float(1 - lam), device=device, dtype=dtype
        )
        one_hot.scatter_add_(1, label1.view(-1, 1).long(), lam1)
        one_hot.scatter_add_(1, label2.view(-1, 1).long(), lam2)
    else:
        one_hot = torch.zeros(label.size(0), num_classes, device=device, dtype=dtype)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
    return one_hot


def get_incremental_margin(m_max, m_min=0, n_epochs=10, mode="linear"):

    if isinstance(m_max, dict):
        m_vals = []
        m_classes = {}
        for class_id, m_class in m_max.items():
            if mode == "linear":
                m_classes[class_id] = np.linspace(m_min, m_class, n_epochs)
            if mode == "log":
                m_classes[class_id] = np.logspace(m_min, m_class, n_epochs)
        for e_i in range(n_epochs):
            e_m = {}
            for class_id, class_m_vals in m_classes.items():
                e_m[class_id] = class_m_vals[e_i]
            m_vals.append(e_m)
    else:
        if mode == "linear":
            m_vals = np.linspace(m_min, m_max, n_epochs)
        if mode == "log":
            m_vals = np.logspace(m_min, m_max, n_epochs)

    return m_vals
