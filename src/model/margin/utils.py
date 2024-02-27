import torch
import numpy as np

def l2_norm(input_x,axis=1):
    norm = torch.norm(input_x,2,axis,True)
    output = torch.div(input_x, norm)
    return output


def get_incremental_margin(
        m_max,
        m_min=0,
        n_epochs=10, 
        mode='linear'
    ):
    
    if isinstance(m_max, dict):
        m_vals = []
        m_classes = {}
        for class_id, m_class in m_max.items():
            if mode=='linear':
                m_classes[class_id] = np.linspace(m_min, m_class, n_epochs)
            if mode=='log':
                m_classes[class_id] = np.logspace(m_min, m_class, n_epochs)
        for e_i in n_epochs:
            e_m = {}
            for class_id, m_vals in m_classes.items():
                e_m[class_id] = m_vals[e_i]
            m_vals.append(e_m)
    else:
        if mode=='linear':
            m_vals = np.linspace(m_min, m_max, n_epochs)
        if mode=='log':
            m_vals = np.logspace(m_min, m_max, n_epochs)

    return m_vals