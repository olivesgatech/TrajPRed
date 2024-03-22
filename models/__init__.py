from models.trajpredict.relation.tprn_re_np import TrajPRed_re_NP
from models.trajpredict.relation.tprn_np_sdd import TrajPRed_NP_SDD


_MODELS_ = {
    'tprn_re_np': TrajPRed_re_NP,
    'tprn_np_sdd': TrajPRed_NP_SDD,
}

def make_model(cfg):
    model = _MODELS_[cfg.method]
    try:
        return model(cfg)
    except Exception as e:
        print(e)
        return model(cfg.model)
