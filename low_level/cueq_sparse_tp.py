import numpy as np
import jax
import jax.numpy as jnp


def _as_numpy_int(array):
    return np.asarray(array, dtype=np.int64)


def _as_numpy_float(array):
    return np.asarray(array)


def build_sparse_cg_polynomial(
    *,
    index_i1,
    index_i2,
    ens_cg,
    index_den,
    index_add,
    index_squ,
    num_input_orbitals,
    num_iter_orbitals,
    num_output_orbitals,
    num_cg,
):
    import cuequivariance as cue

    index_i1 = _as_numpy_int(index_i1)
    index_i2 = _as_numpy_int(index_i2)
    index_den = _as_numpy_int(index_den)
    index_add = _as_numpy_int(index_add)
    index_squ = _as_numpy_int(index_squ)
    ens_cg = _as_numpy_float(ens_cg)

    descriptor = cue.SegmentedTensorProduct.from_subscripts("i,j,k,l+ijkl")
    descriptor.add_segments(0, [(1,)] * int(num_input_orbitals))
    descriptor.add_segments(1, [(1,)] * int(num_iter_orbitals))
    descriptor.add_segments(2, [(1,)] * int(num_cg))
    descriptor.add_segments(3, [(1,)] * int(num_output_orbitals))

    for path_id, cg_value in enumerate(ens_cg):
        den_id = index_den[path_id]
        out_id = index_add[den_id]
        coeff_id = index_squ[den_id]
        descriptor.add_path(
            int(index_i1[path_id]),
            int(index_i2[path_id]),
            int(coeff_id),
            int(out_id),
            c=np.asarray(cg_value).reshape(1, 1, 1, 1),
        )

    stp = descriptor.consolidate_paths()
    return cue.SegmentedPolynomial(
        inputs=[stp.operands[0], stp.operands[1], stp.operands[2]],
        outputs=[stp.operands[3]],
        operations=[(cue.Operation([0, 1, 2, 3]), stp)],
    )


def sparse_cg_tensor_product_cueq(
    polynomial,
    init_orb,
    iter_orb,
    l_coeff,
    *,
    num_output_orbitals,
    method="naive",
):
    import cuequivariance_jax as cuex

    dtype = init_orb.dtype
    num_nodes, _, num_waves = init_orb.shape
    batch_size = num_nodes * num_waves

    init_flat = init_orb.transpose(0, 2, 1).reshape(batch_size, init_orb.shape[1])
    iter_flat = iter_orb.transpose(0, 2, 1).reshape(batch_size, iter_orb.shape[1])
    coeff_flat = l_coeff.transpose(1, 2, 0).reshape(batch_size, l_coeff.shape[0])

    [out_flat] = cuex.segmented_polynomial(
        polynomial,
        [init_flat, iter_flat, coeff_flat],
        [jax.ShapeDtypeStruct((batch_size, int(num_output_orbitals)), dtype)],
        method=method,
        math_dtype=jnp.dtype(dtype).name,
    )
    return out_flat.reshape(num_nodes, num_waves, int(num_output_orbitals)).transpose(2, 0, 1)
