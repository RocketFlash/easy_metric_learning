from .model import MLNet, get_model_embeddings
from .margin import get_margin


def get_model(
        backbone_config,
        head_config,
        margin_config=None,
        n_classes=1000
    ):
        
    embeddings_model = get_model_embeddings(
        backbone_config=backbone_config,
        head_config=head_config
    )
    
    if margin_config is not None:
        embeddings_size = head_config.embeddings_size
        margin = get_margin(
            margin_config=margin_config,
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