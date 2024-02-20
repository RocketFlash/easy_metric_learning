from .adacos import AdaCos
from .adaface import AdaFace
from .arcface import ArcMarginProduct, AddMarginProduct
from .cosface import CosFace
from .sphereface import SphereFace
from .subcenter_arcface import SubcenterArcMarginProduct
from .elasticface import ElasticArcFace, ElasticCosFace
from .curricularface import CurricularFace
from .softmax import Softmax
from .combined import CombinedMargin


def get_margin(
        margin_config,
        embeddings_size=512, 
        n_classes=100,
    ):

    margin_type=margin_config.type

    if margin_type=='adacos':
        margin = AdaCos(
            in_features=embeddings_size,
            out_features=n_classes, 
            m=margin_config.m,  
            ls_eps=margin_config.ls_eps
        )
    elif margin_type=='adaface_bn':
        margin = AdaFace(
            in_features=embeddings_size,
            out_features=n_classes,
            m=margin_config.m,
            h=margin_config.h,
            s=margin_config.s,
            t_alpha=margin_config.t_alpha,
            ls_eps=margin_config.ls_eps,
            use_batchnorm=True
        )
    elif margin_type=='adaface':
        margin = AdaFace(
            in_features=embeddings_size,
            out_features=n_classes,
            m=margin_config.m,
            h=margin_config.h,
            s=margin_config.s,
            t_alpha=margin_config.t_alpha
        )
    elif margin_type=='cosface':
        margin = AddMarginProduct(
            in_features=embeddings_size,
            out_features=n_classes, 
            s=margin_config.s, 
            m=margin_config.m
        )
    elif margin_type=='subcenter_arcface':
        margin = SubcenterArcMarginProduct(
            in_features=embeddings_size,
            out_features=n_classes, 
            s=margin_config.s, 
            m=margin_config.m,
            K=margin_config.K, 
            easy_margin=margin_config.easy_margin, 
            ls_eps=margin_config.ls_eps
        )
    elif margin_type=='elastic_arcface':
        margin = ElasticArcFace(
            in_features=embeddings_size,
            out_features=n_classes, 
            s=margin_config.s, 
            m=margin_config.m, 
            plus=margin_config.plus
        )
    elif margin_type=='elastic_arcface_plus':
        margin = ElasticArcFace(
            in_features=embeddings_size,
            out_features=n_classes, 
            s=margin_config.s, 
            m=margin_config.m, 
            plus=margin_config.plus
        )
    elif margin_type=='elastic_cosface':
        margin = ElasticCosFace(
            in_features=embeddings_size,
            out_features=n_classes, 
            s=margin_config.s, 
            m=margin_config.m, 
            plus=margin_config.plus
        )
    elif margin_type=='elastic_cosface_plus':
        margin = ElasticCosFace(
            in_features=embeddings_size,
            out_features=n_classes, 
            s=margin_config.s, 
            m=margin_config.m, 
            plus=margin_config.plus
        )
    elif margin_type=='arcface':
        margin = ArcMarginProduct(
            in_features=embeddings_size,
            out_features=n_classes, 
            s=margin_config.s, 
            m=margin_config.m, 
            easy_margin=margin_config.easy_margin, 
            ls_eps=margin_config.ls_eps
        )
    elif margin_type=='curricularface':
        margin = CurricularFace(
            in_features=embeddings_size,
            out_features=n_classes, 
            s=margin_config.s, 
            m=margin_config.m,  
            ls_eps=margin_config.ls_eps
        )
    elif margin_type=='combined':
        margin = CombinedMargin(
            in_features=embeddings_size,
            out_features=n_classes,
            s=margin_config.s,
            m1=margin_config.m1,
            m=margin_config.m,
            m3=margin_config.m3,
            ls_eps=margin_config.ls_eps,  
            easy_margin=margin_config.easy_margin
        )
    else:
        margin = Softmax(
            in_features=embeddings_size,
            out_features=n_classes
        )
    
    return margin