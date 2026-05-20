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
from .magface import MagFace
from .circle import CircleMargin
from .partialfc import DistributedPartialFCArcMarginProduct, PartialFCArcMarginProduct
from .x2_softmax import X2Softmax
from .qamface import QAMFace
import numpy as np


def calculate_dynamic_margin(
    dynamic_margin_config,
    id_counts,
):
    dynamic_margin = {}
    for class_id, class_id_cnt in id_counts.items():
        dynamic_margin[class_id] = (
            dynamic_margin_config.hb * class_id_cnt ** (-dynamic_margin_config.lmbda)
            + dynamic_margin_config.lb
        )

    return dynamic_margin


def calculate_autoscale(n_classes):
    return np.sqrt(2) * np.log(n_classes - 1)


def get_margin(
    config_margin,
    embeddings_size=512,
    n_classes=100,
):

    margin_type = config_margin.type
    if margin_type == "softmax":
        return Softmax(in_features=embeddings_size, out_features=n_classes)

    if config_margin.autoscale:
        s = calculate_autoscale(n_classes)
    else:
        s = config_margin.s

    if margin_type == "magface":
        return MagFace(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            l_a=config_margin.l_a,
            u_a=config_margin.u_a,
            l_margin=config_margin.l_margin,
            u_margin=config_margin.u_margin,
            lambda_g=config_margin.lambda_g,
            easy_margin=config_margin.easy_margin,
            ls_eps=config_margin.ls_eps,
        )

    if margin_type == "qamface":
        return QAMFace(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            min_s=config_margin.min_s,
            l_a=config_margin.l_a,
            u_a=config_margin.u_a,
            l_margin=config_margin.l_margin,
            u_margin=config_margin.u_margin,
            easy_margin=config_margin.easy_margin,
            ls_eps=config_margin.ls_eps,
        )

    if config_margin.dynamic_margin is not None:
        m = calculate_dynamic_margin(
            config_margin.dynamic_margin,
            config_margin.id_counts,
        )
    else:
        m = config_margin.m

    if margin_type == "adacos":
        margin = AdaCos(
            in_features=embeddings_size,
            out_features=n_classes,
            m=m,
            ls_eps=config_margin.ls_eps,
        )
    elif margin_type == "adaface":
        margin = AdaFace(
            in_features=embeddings_size,
            out_features=n_classes,
            m=m,
            h=config_margin.h,
            s=s,
            t_alpha=config_margin.t_alpha,
            ls_eps=config_margin.ls_eps,
            use_batchnorm=config_margin.use_batchnorm,
        )
    elif margin_type == "cosface":
        margin = AddMarginProduct(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            m=m,
            ls_eps=config_margin.ls_eps,
        )
    elif margin_type == "circle":
        margin = CircleMargin(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            m=m,
            ls_eps=config_margin.ls_eps,
        )
    elif margin_type == "x2_softmax":
        margin = X2Softmax(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            m=m,
            min_m=config_margin.min_m,
            easy_margin=config_margin.easy_margin,
            ls_eps=config_margin.ls_eps,
        )
    elif margin_type == "subcenter_arcface":
        margin = SubcenterArcMarginProduct(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            m=m,
            K=config_margin.K,
            easy_margin=config_margin.easy_margin,
            ls_eps=config_margin.ls_eps,
        )
    elif margin_type == "partialfc_arcface":
        margin = PartialFCArcMarginProduct(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            m=m,
            sample_rate=config_margin.sample_rate,
            min_sample_classes=config_margin.min_sample_classes,
            easy_margin=config_margin.easy_margin,
            ls_eps=config_margin.ls_eps,
        )
    elif margin_type == "distributed_partialfc_arcface":
        margin = DistributedPartialFCArcMarginProduct(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            m=m,
            sample_rate=config_margin.sample_rate,
            min_sample_classes=config_margin.min_sample_classes,
            easy_margin=config_margin.easy_margin,
            ls_eps=config_margin.ls_eps,
        )
    elif margin_type == "elastic_arcface":
        margin = ElasticArcFace(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            m=m,
            plus=config_margin.plus,
        )
    elif margin_type == "elastic_arcface_plus":
        margin = ElasticArcFace(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            m=m,
            plus=config_margin.plus,
        )
    elif margin_type == "elastic_cosface":
        margin = ElasticCosFace(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            m=m,
            plus=config_margin.plus,
        )
    elif margin_type == "elastic_cosface_plus":
        margin = ElasticCosFace(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            m=m,
            plus=config_margin.plus,
        )
    elif margin_type == "arcface":
        margin = ArcMarginProduct(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            m=m,
            easy_margin=config_margin.easy_margin,
            ls_eps=config_margin.ls_eps,
        )
    elif margin_type == "curricularface":
        margin = CurricularFace(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            m=m,
            ls_eps=config_margin.ls_eps,
        )
    elif margin_type == "sphereface":
        margin = SphereFace(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            m=m,
            ls_eps=config_margin.ls_eps,
        )
    elif margin_type == "combined":
        margin = CombinedMargin(
            in_features=embeddings_size,
            out_features=n_classes,
            s=s,
            m1=config_margin.m1,
            m=m,
            m3=config_margin.m3,
            ls_eps=config_margin.ls_eps,
            easy_margin=config_margin.easy_margin,
        )
    else:
        raise ValueError(f"Unknown margin type: {margin_type}")

    return margin
