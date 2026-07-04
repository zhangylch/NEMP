from flax import nnx
import jax
import jax.numpy as jnp
import math
from typing import Optional


class ScaledDense(nnx.Module):
    """
    A Dense layer with custom scaling.
    """

    def __init__(
        self,
        in_features: int,
        features: int,
        cst: float = 1.0,
        use_bias: bool = False,
        bias_init_value: Optional[jnp.ndarray] = None,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.features = features
        self.use_bias = use_bias
        self.dtype = dtype

        scale = cst / math.sqrt(in_features)
        if use_bias:
            scale = scale / 1e2
        self.scale = float(scale)

        self.kernel = nnx.Param(
            nnx.initializers.normal(1.0)(rngs.params(), (in_features, features), dtype)
        )

        if use_bias:
            if bias_init_value is not None:
                assert bias_init_value.shape == (features,), (
                    f"bias_init_value shape mismatch, expected {(features,)}, "
                    f"got {bias_init_value.shape}"
                )
                bias = bias_init_value.astype(dtype)
            else:
                bias = jnp.zeros(features, dtype=dtype)
            self.bias = nnx.Param(bias)

    def __call__(self, x):
        assert x.shape[-1] == self.in_features, (
            f"Input shape {x.shape} does not match layer's in_features {self.in_features}"
        )

        out = x @ (self.kernel[...] * jnp.array(self.scale, dtype=self.dtype))
        if self.use_bias:
            out += self.bias[...]
        return out


class ResidualBlock(nnx.Module):
    """A self-contained residual block module."""

    def __init__(
        self,
        features: int,
        layers_per_block: int,
        cst: float,
        scale: float,
        use_bias: bool,
        dtype: jnp.dtype,
        *,
        rngs: nnx.Rngs,
    ):
        self.features = features
        self.layers_per_block = layers_per_block
        self.residual_scale = float(scale)
        self.dtype = dtype

        self.layers = nnx.List(
            [
                ScaledDense(
                    in_features=features,
                    features=features,
                    cst=cst,
                    use_bias=use_bias,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(layers_per_block)
            ]
        )

    def __call__(self, x):
        residual = x

        for layer in self.layers:
            x = jax.nn.silu(x)
            x = layer(x)

        residual_scale = jnp.array(self.residual_scale, dtype=self.dtype)
        residual_norm = jnp.sqrt(
            jnp.array(1.0, dtype=self.dtype) + residual_scale * residual_scale
        )
        x = (x + residual * residual_norm) / residual_norm
        return x


class MLP(nnx.Module):
    """
    A Multi-Layer Perceptron using Flax NNX.
    """

    def __init__(
        self,
        in_features: int,
        num_output: int = 1,
        num_blocks: int = 1,
        features: int = 128,
        layers_per_block: int = 2,
        cst: float = 1.0,
        use_bias: bool = False,
        use_linear: bool = False,
        bias_init_value: Optional[jnp.ndarray] = None,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.num_output = num_output
        self.num_blocks = num_blocks
        self.features = features
        self.layers_per_block = layers_per_block
        self.use_linear = use_linear
        self.dtype = dtype

        if not use_linear:
            self.input_layer = ScaledDense(
                in_features=in_features,
                features=features,
                cst=cst,
                use_bias=use_bias,
                dtype=dtype,
                rngs=rngs,
            )

            self.blocks = nnx.List(
                [
                    ResidualBlock(
                        features=features,
                        layers_per_block=layers_per_block,
                        cst=cst,
                        scale=float(num_blocks),
                        use_bias=use_bias,
                        dtype=dtype,
                        rngs=rngs,
                    )
                    for _ in range(num_blocks)
                ]
            )

            output_in_features = features
        else:
            output_in_features = in_features

        self.output_layer = ScaledDense(
            in_features=output_in_features,
            features=num_output,
            cst=1.0,
            use_bias=use_bias,
            bias_init_value=bias_init_value,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        if not self.use_linear:
            x = self.input_layer(x)
            for block in self.blocks:
                x = block(x)
            x = jax.nn.silu(x)

        return self.output_layer(x)
