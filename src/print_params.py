from collections.abc import Mapping
from flax import nnx


def _array_leaf(value):
    if isinstance(value, nnx.Variable):
        return value[...]
    return value


def print_params(params, parent_key=""):
    if isinstance(params, Mapping) and "params" in params and len(params) == 1:
        params = params["params"]

    if isinstance(params, nnx.State):
        params = nnx.to_pure_dict(params)

    if isinstance(params, Mapping):
        for key, value in params.items():
            full_key = f"{parent_key}/{key}" if parent_key else str(key)
            print_params(value, full_key)
        return

    value = _array_leaf(params)
    if hasattr(value, "shape"):
        print(f"{parent_key}: {value.shape}")
