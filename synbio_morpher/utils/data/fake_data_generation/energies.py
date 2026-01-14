

import numpy as np
from functools import partial
from synbio_morpher.utils.misc.numerical import make_symmetrical_matrix_from_sequence
import jax
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt

RNA_EU = 0
RNA_EL = -1.5


def generate_energies(n_circuits: int, n_species: int, len_seq: int, p_null: float,
                      seed: int, symmetrical=True, type_energies: str = 'RNA') -> np.ndarray:
    """
    Generate a list of interaction energies for n_circuits circuits with n_species species each.
    The sequence length (len_seq) is the approximate length of the interacting components if they existed.
    For example, for RNA of length 20, the strongest interaction energy is different to RNAs of length 100.
    """
    rng = jax.random.PRNGKey(seed)
    fn = partial(transform_to_rna_energies, 
                 rng=rng, len_seq=len_seq,
                 p_null=p_null) if type_energies == 'RNA' else lambda x: x
    if symmetrical:
        n_interacting = int(np.sum(np.arange(n_species + 1)))
        energies = fn(np.array(jax.random.uniform(
            rng, (n_circuits, n_interacting))))
        energies = np.array(jax.vmap(partial(
            make_symmetrical_matrix_from_sequence, side_length=n_species))(energies))
    else:
        energies = fn(jax.random.uniform(rng, shape=(n_circuits, n_species, n_species)))
    return energies


def pepper_noninteracting(rng, energies: np.ndarray, p_null: float) -> np.ndarray:
    """ p_null is the percentage of non-interacting species """
    mask = np.array(jax.random.choice(rng, jnp.arange(energies.size), shape=(int(energies.size * p_null),), replace=False))
    energies.flat[mask] = 0
    return energies


def transform_to_rna_energies(rnd: np.ndarray, rng, len_seq: int, p_null: float) -> np.ndarray:
    """
    Transform interaction energies to RNA interaction energies.
    """
    rnd = np.log(rnd + 5e-2)
    energies = np.interp(rnd, (rnd.min(), rnd.max()),
                         (RNA_EL*len_seq, RNA_EU*len_seq))
    energies = pepper_noninteracting(rng, energies, p_null=p_null)
    return energies


def main():
    e = generate_energies(n_circuits=3000, n_species=3, len_seq=20, p_null=0.1,
                          symmetrical=True, type_energies='RNA', seed=0)
    sns.histplot(e[:, np.triu_indices(3)].flatten(), bins=50, element='step')
    plt.savefig('test.png')
