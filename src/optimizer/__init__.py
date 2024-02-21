import hydra


def get_optimizer(model, optimizer_config):
    params = model.parameters()
    
    if optimizer_config.backbone_lr_scaler<1:
        params = [
            {
                'params': model.embeddings_net.backbone.parameters(),
                'lr':     optimizer_config.optimizer.lr * optimizer_config.backbone_lr_scaler
            },
            {
                'params': model.embeddings_net.head.parameters(),
                'lr':     optimizer_config.optimizer.lr
            },
            {
                'params': model.margin.parameters(),
                'lr':     optimizer_config.optimizer.lr
            }
        ]

    loss_fn = hydra.utils.instantiate(
        optimizer_config.get('optimizer'), 
        params=params,
        _convert_="object"
    )
    
    return loss_fn