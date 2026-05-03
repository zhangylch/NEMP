import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.ops import segment_sum

from low_level import cg_cal
from low_level import cueq_sparse_tp


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cuequivariance") is None
    or importlib.util.find_spec("cuequivariance_jax") is None,
    reason="cuequivariance and cuequivariance_jax are required",
)


def _contract_sph(rmaxl, prmaxl):
    index_i1 = []
    index_i2 = []
    cg_array = []
    index_den = []
    index_squ = []
    index_add = []
    count_l = np.zeros(prmaxl)
    num_coeff = 0
    num_den = 0

    for lf in range(prmaxl):
        for li1 in range(rmaxl):
            low = abs(li1 - lf)
            up = min(prmaxl, li1 + lf + 1)
            for li2 in range(low, up):
                if np.mod(li1 + li2 + lf, 2) < 0.5:
                    cg = cg_cal.clebsch_gordan(li1, li2, lf)
                    count_l[lf] += 1
                    for mf in range(0, 2 * lf + 1):
                        dim3 = lf * lf + mf
                        index_add.append(dim3)
                        index_squ.append(num_coeff)
                        for mi1 in range(0, 2 * li1 + 1):
                            for mi2 in range(0, 2 * li2 + 1):
                                dim1 = li1 * li1 + mi1
                                dim2 = li2 * li2 + mi2
                                if np.abs(cg[mi1, mi2, mf]) > 1e-3:
                                    index_i1.append(dim1)
                                    index_i2.append(dim2)
                                    cg_array.append(cg[mi1, mi2, mf])
                                    index_den.append(num_den)
                        num_den += 1
                    num_coeff += 1

    return {
        "index_i1": jnp.asarray(index_i1),
        "index_i2": jnp.asarray(index_i2),
        "ens_cg": jnp.asarray(cg_array),
        "index_add": jnp.asarray(index_add),
        "index_den": jnp.asarray(index_den),
        "index_squ": jnp.asarray(index_squ),
        "count_l": jnp.asarray(count_l),
        "num_cg": num_coeff,
    }


def _custom_sparse_tp(paths, init_orb, iter_orb, l_coeff, num_output_orbitals):
    inter_orbital = jnp.einsum(
        "ikj, ikj, k -> kij",
        init_orb[:, paths["index_i1"]],
        iter_orb[:, paths["index_i2"]],
        paths["ens_cg"],
    )
    mp_orbital = segment_sum(
        inter_orbital,
        paths["index_den"],
        num_segments=paths["index_add"].shape[0],
        indices_are_sorted=True,
    )
    return segment_sum(
        mp_orbital * l_coeff[paths["index_squ"]],
        paths["index_add"],
        num_segments=num_output_orbitals,
    )


def test_cueq_sparse_tp_matches_custom_lmax2():
    rmaxl = 3
    prmaxl = 3
    num_nodes = 4
    num_waves = 5
    num_input_orbitals = rmaxl * rmaxl
    num_output_orbitals = prmaxl * prmaxl

    paths = _contract_sph(rmaxl, prmaxl)
    key = jax.random.PRNGKey(0)
    key_init, key_iter, key_coeff = jax.random.split(key, 3)

    init_orb = jax.random.normal(key_init, (num_nodes, num_input_orbitals, num_waves), dtype=jnp.float32)
    iter_orb = jax.random.normal(key_iter, (num_nodes, num_output_orbitals, num_waves), dtype=jnp.float32)
    l_coeff = jax.random.normal(key_coeff, (paths["num_cg"], num_nodes, num_waves), dtype=jnp.float32)

    custom = _custom_sparse_tp(paths, init_orb, iter_orb, l_coeff, num_output_orbitals)
    polynomial = cueq_sparse_tp.build_sparse_cg_polynomial(
        index_i1=paths["index_i1"],
        index_i2=paths["index_i2"],
        ens_cg=paths["ens_cg"],
        index_den=paths["index_den"],
        index_add=paths["index_add"],
        index_squ=paths["index_squ"],
        num_input_orbitals=num_input_orbitals,
        num_iter_orbitals=num_output_orbitals,
        num_output_orbitals=num_output_orbitals,
        num_cg=paths["num_cg"],
    )
    cueq = cueq_sparse_tp.sparse_cg_tensor_product_cueq(
        polynomial,
        init_orb,
        iter_orb,
        l_coeff,
        num_output_orbitals=num_output_orbitals,
    )

    assert cueq.shape == custom.shape
    assert cueq.dtype == custom.dtype
    np.testing.assert_allclose(np.asarray(cueq), np.asarray(custom), rtol=1e-5, atol=1e-6)
