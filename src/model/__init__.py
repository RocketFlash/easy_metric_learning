import omegaconf
from .model import MLNet, get_model_embeddings
from .margin import get_margin
from ..utils import load_checkpoint


def get_model(
        config_backbone,
        config_head,
        config_margin=None,
        n_classes=1000
    ):
        
    embeddings_model = get_model_embeddings(
        config_backbone=config_backbone,
        config_head=config_head
    )
    
    if config_margin is not None:
        if config_head.type=='no_head':
            config_head.embeddings_size = embeddings_model.backbone_out_feats
            
        embeddings_size = config_head.embeddings_size
        margin = get_margin(
            config_margin=config_margin,
            embeddings_size=embeddings_size,
            n_classes=n_classes
        )
        
        model = MLNet(
            embeddings_net=embeddings_model, 
            margin=margin
        )
    else:
        model = embeddings_model
    
    return model


def get_model_teacher(
        config_teacher,
        device=None,
        accelerator=None
    ):
        
    config = omegaconf.OmegaConf.load(config_teacher.model.config)
    
    model_teacher = load_emb_model_and_weights(
        config.backbone,
        config.head,
        weights=config_teacher.model.weights,
        device=device,
        accelerator=accelerator
    )

    model_teacher.eval()
    return model_teacher


def load_emb_model_and_weights(
        config_backbone,
        config_head,
        weights=None,
        device=None,
        accelerator=None
    ):
    model = get_model(
        config_backbone=config_backbone,
        config_head=config_head,
    )

    if weights is not None:
        checkpoint_data = load_checkpoint(
            weights, 
            model=model, 
            mode='emb',
            device=device,
            accelerator=accelerator
        )
        model = checkpoint_data['model'] if 'model' in checkpoint_data else model

    return model