from flax import linen as nn
import jax.numpy as jnp
from typing import Optional
import jax

class ScaledDense(nn.Module):
    features: int
    cst: float = 1.0
    use_bias: bool = False
    bias_init_value: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self, x):
        
        # 权重初始化逻辑
        kernel = self.param(
            'kernel',
            nn.initializers.normal(1.0),  # 固定标准差为1.0
            (x.shape[-1], self.features)
        )
        
        scale = jnp.array(self.cst) / jnp.sqrt(x.shape[-1])
        if self.use_bias:
            scale = scale / jnp.array(1e2)

        
        # 执行线性变换
        out = x @ (kernel * scale)
        #jax.debug.print("dtype, {x} {y}", x=kernel.dtype, y=out.dtype)

        if self.use_bias:
            def bias_init_fn(rng):
                if self.bias_init_value is not None:
                    # 验证传入的初始值形状是否正确
                    assert self.bias_init_value.shape == (self.features,), \
                        f"bias_init_value shape mismatch, expected {(self.features,)}, got {self.bias_init_value.shape}"
                    return self.bias_init_value
                return jnp.zeros(self.features)
            
            bias = self.param('bias', bias_init_fn)
            out += bias
            
        return out

class MLP(nn.Module):
    num_output: int = 1
    num_blocks: int = 1
    features: int = 128
    layers_per_block: int = 2
    cst: float = 1.0
    use_bias: bool = False
    use_linear: bool = False
    bias_init_value: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self, x):
        if not self.use_linear:
            # 初始投影层
            x = ScaledDense(
                features=self.features,
                cst = self.cst,
                use_bias=self.use_bias,
                bias_init_value=None,
                name=f'input_layer'
            )(x)
            
            # 残差块处理
            for block_idx in range(self.num_blocks):
                residual = x
                #x = nn.LayerNorm(use_bias=self.use_bias)(x)
                for layer_idx in range(self.layers_per_block):
                    x = nn.silu(x)
                    x = ScaledDense(
                        features=self.features,
                        cst = self.cst,
                        use_bias=self.use_bias,
                        bias_init_value=None,
                        name=f'block_{block_idx}_layer_{layer_idx}'
                    )(x)
                    
                # 残差连接
                x = (x + residual) / jnp.sqrt(2.0)
            
            x = nn.silu(x)
            x = ScaledDense(
                features=self.num_output,
                cst = 1.0,
                use_bias=self.use_bias,
                bias_init_value=self.bias_init_value,
                name=f'output_layer'
            )(x)
        else:
            x = ScaledDense(
                features=self.features,
                cst = 1.0,
                use_bias=self.use_bias,
                bias_init_value=None,
                name=f'input_layer'
            )(x)
        
            x = ScaledDense(
                features=self.num_output,
                cst = 1.0,
                use_bias=self.use_bias,
                bias_init_value=self.bias_init_value,
                name=f'output_layer'
            )(x)
        
        return x
