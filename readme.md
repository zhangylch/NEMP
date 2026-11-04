# NEMP
## Introduction:
This repository provides an implementation of Node-Equivariant Message Passing Neural Networks (NEMP) for representing potential energy surfaces, built on the JAX framework. NEMP achieves the accuracy of state-of-the-art equivariant message-passing (MP) models while retaining the efficiency of local descriptor-based approaches by reformulating equivariant message passing along the node space rather than the edge space.

Molecular dynamics (MD) simulations can be performed using the JAX-MD package to fully leverage GPU acceleration and achieve optimal performance. For more details about the methodology and benchmarks, please refer to the associated paper.
## Requirements:
* jax+flax
* optax
* ASE
* JAX-MD

## Examples:
Training can be performed by running the following command:
```
python3 $path "train"
```
where $path is the directory where the code is located. This command executes the train.py script inside the train folder.

All training parameters are specified in the config.json file, and the accepted input data format is extended XYZ. Examples of config.json are provided in the examples folder.

## NNX model interface

Training, inference, and ASE models each expose one `MPNN(nnx.Module)` class.
Construct the model with an explicit RNG and call it directly:

```python
from flax import nnx
from train_model.MPNN import MPNN

model = MPNN(config, rngs=nnx.Rngs(0))
total_energy, graph_energy = model(*data)
```

Training and evaluation keep explicit parameter state for JAX/Optax: initialize
with `graphdef, params = nnx.split(model, nnx.Param)`, then call
`nnx.merge(graphdef, params)(*data)` inside the energy function. The old
`MPNNCore` and Linen-style `model.init` / `model.apply` wrapper are removed.
Checkpoint mappings are converted once at load time with
`checkpoint_params(saved_params, template_params)` from `src.data_config`.
Parameter names, shapes, initialization order, and model equations are unchanged
from the preceding `nnx` implementation.

The single-graph inference and ASE models return a scalar energy, take an
unbatched cell, omit `celllist`, and use shifts shaped `(nedge, 3)`. The training
model takes batched cells and shifts shaped `(3, nedge)`.

## References
If you use this package, please cite these works.
1. NEMP model: Yaolong Zhang and Hua Guo. [abs/2508.16086](https://arxiv.org/abs/2508.16086)
