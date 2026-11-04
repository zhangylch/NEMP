from dataclasses import asdict
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase import Atoms
from ase.stress import full_3x3_to_voigt_6_stress
from flax import nnx

import ASE.NEMP as ase_nemp
from inference_model.MPNN import MPNN
from test_nnx_mpnn_smoke import _minimal_config


class _TwoAtomNeighbors:
    """Supply two directed edges and zero-distance padding in Fortran layout."""

    def init_neigh(self, *args):
        pass

    def get_neigh(self, positions, capacity):
        neighbors = np.zeros((2, int(capacity)), dtype=np.int32)
        neighbors[:, :2] = [[0, 1], [1, 0]]
        shifts = np.zeros((3, int(capacity)), dtype=positions.dtype)
        return positions.copy(), neighbors, shifts, 2


@pytest.fixture(scope="module")
def ase_reference():
    config = _minimal_config()
    graphdef, ema_params = nnx.split(MPNN(config, rngs=nnx.Rngs(7)), nnx.Param)
    _, raw_params = nnx.split(MPNN(config, rngs=nnx.Rngs(42)), nnx.Param)
    positions = jnp.array([[0.1, 0.2, 0.3], [1.2, 0.4, -0.2]], dtype=jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32) * 8.0
    neighbors = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    shifts = jnp.zeros((2, 3), dtype=jnp.float32)
    factors = jnp.ones(2, dtype=jnp.float32)
    species = jnp.ones(2, dtype=jnp.int32)

    def energy(coordinates, strain):
        return nnx.merge(graphdef, ema_params)(
            coordinates, cell, strain, neighbors, shifts, factors, species
        )

    value, (coordinate_grad, strain_grad) = jax.value_and_grad(energy, (0, 1))(
        positions, jnp.zeros_like(cell)
    )
    return config, raw_params, ema_params, positions, cell, value, coordinate_grad, strain_grad


@pytest.mark.parametrize("checkpoint_format", ["state", "plain"])
@pytest.mark.parametrize(
    "properties", [["energy"], ["energy", "forces"], ["energy", "forces", "stress"]]
)
def test_ase_calculator_uses_ema_and_preserves_derivative_signs(
    monkeypatch, ase_reference, checkpoint_format, properties
):
    config, raw_params, ema_params, positions, cell, energy, coordinate_grad, strain_grad = ase_reference
    atomic_energy = 0.7
    full_config = SimpleNamespace(
        initpot=atomic_energy,
        jnp_dtype="float32",
        ckpath="unused-test-checkpoint",
        cutoff=config.cutoff,
        maxneigh_per_node=1,
        tp_method=config.tp_method,
        tp_mode=config.tp_mode,
    )
    checkpoint_ema = ema_params if checkpoint_format == "state" else nnx.to_pure_dict(ema_params)
    monkeypatch.setattr(ase_nemp, "load_config", lambda _: full_config)
    monkeypatch.setattr(
        ase_nemp, "restore_checkpoint",
        lambda *_: (3, raw_params, checkpoint_ema, (), asdict(config)),
    )
    atoms = Atoms("H2", positions=np.asarray(positions), cell=np.asarray(cell), pbc=False)
    atoms.calc = ase_nemp.NEMP(
        initatoms=atoms,
        getneigh=_TwoAtomNeighbors(),
        skin_neigh=2,
        properties=properties,
    )

    np.testing.assert_allclose(
        atoms.get_potential_energy(), energy + len(atoms) * atomic_energy,
        rtol=2e-5, atol=2e-5,
    )
    if "forces" in properties:
        assert np.max(np.abs(coordinate_grad)) > 1e-6
        np.testing.assert_allclose(atoms.get_forces(), -coordinate_grad, rtol=2e-5, atol=2e-5)
    if "stress" in properties:
        expected_stress = full_3x3_to_voigt_6_stress(np.asarray(strain_grad) / atoms.get_volume())
        assert np.max(np.abs(expected_stress)) > 1e-8
        np.testing.assert_allclose(atoms.get_stress(), expected_stress, rtol=2e-5, atol=2e-7)
