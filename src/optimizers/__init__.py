import torch.optim as optim
from .sophia_g import SophiaG


def get_optimizer(model, optimizer_config):
    optimizer_type = optimizer_config['OPTIMIZER_TYPE'].lower()

    params = model.parameters()
    lr = float(optimizer_config['LR'])
    weight_decay = float(optimizer_config['WEIGHT_DECAY'])

    if 'BACKBONE_LR_SCALER' in optimizer_config:
        if optimizer_config['BACKBONE_LR_SCALER'] is not None:
            backbone_lr_scaler = float(optimizer_config['BACKBONE_LR_SCALER'])
            if backbone_lr_scaler<1:
                params = []
                params += [{'params': model.embeddings_net.backbone.parameters(),
                                'lr':     lr * backbone_lr_scaler}]
                if model.embeddings_net.pooling is not None:
                    params += [{'params': model.embeddings_net.pooling.parameters(),
                                'lr':     lr}]
                params += [{'params': model.embeddings_net.dropout.parameters(),
                                'lr':     lr}]
                params += [{'params': model.embeddings_net.classifier.parameters(),
                                'lr':     lr}]
                params += [{'params': model.embeddings_net.bn.parameters(),
                                'lr':     lr}]
                params += [{'params': model.margin.parameters(),
                                'lr':     lr}]
                if model.category_net is not None:
                    params += [{'params': model.category_net.parameters(),
                                'lr':     lr}]

    if optimizer_type == 'radam':
        try:
            import torch_optimizer as optim_t
        except ModuleNotFoundError:
            print('torch_optimizer is not installed')
        optimizer = optim_t.RAdam(params,
                                    lr=lr,
                                    betas=(0.9, 0.999),
                                    eps=1e-8,
                                    weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(params,
                                 lr=lr,
                                 weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(params,
                                eps=1e-8,
                                weight_decay=weight_decay)
    elif optimizer_type == 'lion':
        from lion_pytorch import Lion
        optimizer = Lion(params, 
                         lr=lr, 
                         weight_decay=weight_decay)
    elif optimizer_type == 'sophia_g':
        optimizer = SophiaG(params, 
                            lr=lr, 
                            rho=0.04,
                            weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(params,
                                lr=lr,
                                momentum=optimizer_config['MOMENTUM'],
                                nesterov=True,
                                weight_decay=weight_decay)
    return optimizer