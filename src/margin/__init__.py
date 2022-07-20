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

def get_margin(margin_type='arcface',
               embeddings_size=512, 
               out_features=1000,
               s=0.1,
               m=100,
               K=1,
               easy_margin=False,
               ls_eps=0.0):

    if margin_type=='adacos':
        margin = AdaCos(in_features=embeddings_size,
                        out_features=out_features, 
                        m=m,  
                        ls_eps=ls_eps)
    elif margin_type=='adaface_bn':
        margin = AdaFace(in_features=embeddings_size,
                         out_features=out_features,
                         m=m,
                         h=0.333,
                         s=s,
                         t_alpha=1.0,
                         ls_eps=ls_eps,
                         use_batchnorm=True)
    elif margin_type=='adaface':
        margin = AdaFace(in_features=embeddings_size,
                         out_features=out_features,
                         m=m,
                         h=0.333,
                         s=s,
                         t_alpha=1.0)
    elif margin_type=='cosface':
        margin = AddMarginProduct(in_features=embeddings_size,
                                out_features=out_features, 
                                s=s, 
                                m=m)
    elif margin_type=='subcenter_arcface':
        margin = SubcenterArcMarginProduct(in_features=embeddings_size,
                                out_features=out_features, 
                                s=s, 
                                m=m,
                                K=K, 
                                easy_margin=easy_margin, 
                                ls_eps=ls_eps)
    elif margin_type=='elastic_arcface':
        margin = ElasticArcFace(in_features=embeddings_size,
                                out_features=out_features, 
                                s=s, 
                                m=m, 
                                plus=False)
    elif margin_type=='elastic_arcface_plus':
        margin = ElasticArcFace(in_features=embeddings_size,
                                out_features=out_features, 
                                s=s, 
                                m=m, 
                                plus=True)
    elif margin_type=='elastic_cosface':
        margin = ElasticCosFace(in_features=embeddings_size,
                                out_features=out_features, 
                                s=s, 
                                m=m, 
                                plus=False)
    elif margin_type=='elastic_cosface_plus':
        margin = ElasticCosFace(in_features=embeddings_size,
                                out_features=out_features, 
                                s=s, 
                                m=m, 
                                plus=True)
    elif margin_type=='arcface':
        margin = ArcMarginProduct(in_features=embeddings_size,
                                out_features=out_features, 
                                s=s, 
                                m=m, 
                                easy_margin=easy_margin, 
                                ls_eps=ls_eps)
    elif margin_type=='curricularface':
        margin = CurricularFace(in_features=embeddings_size,
                                out_features=out_features, 
                                s=s, 
                                m=m,  
                                ls_eps=ls_eps)
    elif margin_type=='combined':
        margin = CombinedMargin(in_features=embeddings_size,
                                out_features=out_features,
                                s=s,
                                m1=1,
                                m=m,
                                m3=0,
                                ls_eps=ls_eps,  
                                easy_margin=easy_margin)
    else:
        margin = Softmax(in_features=embeddings_size,
                         out_features=out_features)
    
    return margin