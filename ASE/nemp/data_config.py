from jax import vmap, jit
from typing import Sequence, List, Union, Any
from dataclasses import field, dataclass
from jax import Array

#save the arguement for inference 
@dataclass
class ModelConfig:
    ncom_spec: int
    nspec: int
    num_cg: int
    emb_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    MP_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    radial_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    out_nl: Sequence[Union[int, bool]]
    reduce_spec: Any = field(default=None, metadata={'pytree': True})
    com_spec: Any = field(default=None, metadata={'pytree': True}) 
    count_l: Any = field(default=None, metadata={'pytree': True})
    index_l: Any = field(default=None, metadata={'pytree': True})
    index_i1: Any = field(default=None, metadata={'pytree': True})
    index_i2: Any = field(default=None, metadata={'pytree': True})
    ens_cg: Any = field(default=None, metadata={'pytree': True})
    index_add: Any = field(default=None, metadata={'pytree': True})
    index_den: Any = field(default=None, metadata={'pytree': True})
    index_squ: Any = field(default=None, metadata={'pytree': True})
    initbias_neigh: Any = field(default=None, metadata={'pytree': True})
    use_norm: bool=False
    use_bias: bool=False
    cutoff: float = 4.0
    cst: float = 1.67462
    std: float = 1.0
    nwave: int = 16
    nradial: int = 8
    rmaxl: int = 3
    prmaxl: int = 2
    MP_loop: int = 2
    pn: int = 6
    npaircode: int = 8

