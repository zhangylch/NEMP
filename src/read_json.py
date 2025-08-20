
import os
from dataclasses import dataclass, field, replace
from typing import List, Any
import json

@dataclass
class JsonConfig:
    #======general setup===========================================
    local_size: int = field(default=1)
    force_table: bool = field(default=True)
    stress_table: bool = field(default=False)
    stress_sign: int = field(default=1)
    restart: bool = field(default=False)        
    jnp_dtype: str = field(default='float64')   #float32/float64
    batchsize: int = field(default=50) # batchsize for each process
    ncyc: int = field(default=50)
    ntrain: int = field(default=450)
    Fshuffle: bool = field(default=True)
    ene_shift: bool = field(default=True)
    initpot: float = field(default=0.0)
    init_weight: List[float] = field(default_factory = lambda: [1.0, 1.0, 1.0])
    final_weight: List[float] = field(default_factory = lambda: [1.0, 1.0, 1.0])
    queue_size: int = field(default=2)
    use_bias: bool = field(default=False)
    use_norm: bool = field(default=False)
    clip_norm: float = field(default=2.5)
    seed: int = field(default=20)
    pn: int = field(default=2)
    npaircode: int = field(default=16)
    natomcode: int = field(default=8)
    cross_val: bool = field(default=False)
    #========================parameters for optim=======================
    Epoch: int = field(default=20000)                    # total numbers of epochs for fitting 
    warm_lr: float = field(default=0.01)
    warm_epoch: int = field(default=10)
    slr: float = field(default=1e-1) # initial learning rate
    elr: float = field(default=1e-5)                    # final learning rate
    patience_step: int = field(default=50)    
    cooldown: int = field(default=5)
    decay_factor: float = field(default=0.5)
    
    ckpath: str = field(default=os.getcwd()+"/ckpt_dir/")
    ckpath_cpu: str = field(default=os.getcwd()+"/ckpt_dir_cpu/")
    datafolder: str = field(default="/data1/home/hhu17/zyl/data/Equi-EANN/3BPA/date/dataset_3BPA/train_300K.xyz")
    #======= electron sample============
    weight_decay: float = field(default=1e-9)
    
    cutoff: float = field(default=5.0)
    max_l: int = field(default=3)
    pmax_l: int = field(default=2)
    nwave: int = field(default=64)
    nradial: int = field(default=64)
    maxneigh: int = field(default=26)
    MP_loop: int = field(default=3)
    
    #===============================embedded NN structure==========
    emb_nl: List[Any] = field(default_factory = lambda: [2, 128, 2, False])  # nblock, nfeature, nlayer, Layer_norm
    
    radial_nl: List[Any] = field(default_factory = lambda: [1, 128, 2, True])
    MP_nl: List[Any] = field(default_factory = lambda: [1, 256, 0, True])
    out_nl: List[Any] = field(default_factory = lambda: [1, 16, 0, True])  # nblock, nfeature, nlayer, Layer_norm


def load_config(json_path: str) -> JsonConfig:
    default_config = JsonConfig()  # 默认配置
    
    try:
        with open(json_path, "r") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        return default_config
    
    # 仅更新 JSON 中提供的字段
    return replace(default_config, **json_data)

