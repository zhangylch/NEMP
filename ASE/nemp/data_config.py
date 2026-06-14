from jax import vmap, jit
from typing import Sequence, List, Union, Any
from dataclasses import field, dataclass
from jax import Array


def _normalized_tp_value(name, value):
    text = str(value).lower()
    if name == "tp_method" and text in ("uniform1d", "uniform-1d"):
        return "uniform_1d"
    if name == "tp_mode" and text in ("full_mixing", "full-mixing"):
        return "full"
    if name == "tp_mode" and text in ("channel-wise", "channel"):
        return "channelwise"
    return text


def checkpoint_tp_config(model_config, full_config):
    model_config = dict(model_config)
    for name in ("tp_method", "tp_mode"):
        runtime_value = getattr(full_config, name, None)
        checkpoint_value = model_config.get(name, None)
        if checkpoint_value is None:
            if runtime_value is not None:
                model_config[name] = runtime_value
            continue
        if runtime_value is not None and _normalized_tp_value(name, checkpoint_value) != _normalized_tp_value(name, runtime_value):
            raise ValueError(
                f"Checkpoint {name}={checkpoint_value!r} does not match "
                f"full_config {name}={runtime_value!r}. Do not switch TP "
                "backend or mode when loading a trained checkpoint."
            )
    return model_config


#save the arguement for inference 
@dataclass
class ModelConfig:
    nspec: int
    emb_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    MP_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    radial_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    out_nl: Sequence[Union[int, bool]]
    reduce_spec: Any = field(default=None, metadata={'pytree': True})
    com_spec: Any = field(default=None, metadata={'pytree': True})
    index_l: Any = field(default=None, metadata={'pytree': True})
    initbias_neigh: Any = field(default=None, metadata={'pytree': True})
    use_norm: bool=False
    use_bias: bool=False
    cutoff: float = 4.0
    cst: float = 1.67462
    std: float = 1.0
    nwave: int = 16
    npaircode: int = 32
    nradial: int = 8
    rmaxl: int = 3
    prmaxl: int = 2
    MP_loop: int = 2
    pn: int = 6
    tp_method: str = "custom"
    tp_mode: str = "full"
