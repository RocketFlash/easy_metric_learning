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
    ):
        
    config = omegaconf.OmegaConf.load(config_teacher.model.config)
        
    model_teacher = get_model(
        config_backbone=config.backbone,
        config_head=config.head,
    )

    if config_teacher.model.weights is not None:
        checkpoint_data = load_checkpoint(
            config_teacher.model.weights, 
            model=model_teacher, 
            mode='emb'
        )
        model_teacher = checkpoint_data['model'] if 'model' in checkpoint_data else model_teacher

    model_teacher.eval()
    return model_teacher