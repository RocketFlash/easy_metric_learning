import sys
sys.path.append("./")

import torch
from src.evaluator import FacesEvaluator
from src.utils import get_device

if __name__ == '__main__':
    print('Evaluate model on faces')
    val_targets = ['lfw', 'cfp_fp', "agedb_30"]
    rec_prefix = "/home/ubuntu/datasets/metric_learning/ms1mv3"
    model_path = '/home/ubuntu/trained_weights/metric/best_emb_traced.pt'

    device = get_device('')
    backbone = torch.jit.load(model_path, map_location=device).to(device).eval()

    faces_evaluator = FacesEvaluator(val_targets, rec_prefix, device=device)

    print('Start evaluation')
    faces_evaluator(backbone)