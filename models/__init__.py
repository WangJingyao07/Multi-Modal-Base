from .base import MMLBase
from .base_vt import MMLBase_vt
from .base_vta_mmrg import MMRG
from .base_vta_mlu import MLU
from .base_vta import MMLBase_VTA
from .base_vt_vae import MMLBase_av_vae
from .base_vta import MMLBase_VTA
from .base_vta_ib import MMLBase_vta_IB
from .base_vta import MMLBase_VTA
from .base_affection_vta import MMLBase_affection_vta
from .base_affection_va import MMLBase_affection_va
from .bert_vt import BertClf
from .bow_vt import GloveBowClf
from .concat_bert_vt import MultimodalConcatBertClf
from .concat_bow_vt import  MultimodalConcatBowClf
from .image_vt import ImageClf
from .mmbt_vt import MultimodalBertClf
from .late_fusion_vt import MultimodalLateFusionClf
from .tmc_vt import TMC,ce_loss


MODELS = {
    "mml_base": MMLBase_VTA,
    "mml_vt": MMLBase_vt,
    "mml_avt": MMLBase_VTA,
    "mml_av": MMLBase_VTA,
    "mml_avt_mlu": MLU,
    "mml_avt_late": MMLBase_VTA,
    "mml_avt_mmrg": MMRG,
    "mml_avt_b": MMLBase_VTA,
    "mml_av_vae": MMLBase_av_vae,
    "mml_avt_ib": MMLBase_vta_IB,
    "mml_vt_vae": MMLBase_VTA,
    "mml_affection_vta": MMLBase_affection_vta,
    "mml_affection_va": MMLBase_affection_va,
    "mml_avt_mmrgr": MMRG,
    "bert": BertClf,
    "bow": GloveBowClf,
    "concatbow": MultimodalConcatBowClf,
    "concatbert": MultimodalConcatBertClf,
    "img": ImageClf,
    "mmbt": MultimodalBertClf,
    'latefusion':MultimodalLateFusionClf,
    'tmc':TMC
}

def get_model(args):
    return MODELS[args.model](args)
