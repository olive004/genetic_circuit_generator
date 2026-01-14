
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from fire import Fire
from functools import partial
import jax
import jax.numpy as jnp
import os
from typing import List, Optional
from datetime import datetime
import numpy as np
from synbio_morpher.srv.io.manage.script_manager import script_preamble
from synbio_morpher.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from synbio_morpher.utils.common.setup import construct_circuit_from_cfg, prepare_config
from synbio_morpher.utils.results.analytics.naming import get_analytics_types_all, get_true_names_analytics
from synbio_morpher.utils.results.experiments import Experiment, Protocol
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.misc.numerical import make_symmetrical_matrix_from_sequence
from synbio_morpher.utils.parameter_inference.interpolation_grid import create_parameter_range
from synbio_morpher.utils.circuit.agnostic_circuits.circuit_manager import CircuitModeller


# Create matrices
def define_matrices(num_species, size_interaction_array, num_unique_interactions, analytic_types) -> np.ndarray:
    matrix_dimensions = tuple(
        [num_species] + [size_interaction_array]*num_unique_interactions)
    matrix_size = num_species * \
        np.power(size_interaction_array, num_unique_interactions)
    assert matrix_size == np.prod(list(
        matrix_dimensions)), 'Something is off about the intended size of the matrix'

    all_analytic_matrices = np.zeros(
        tuple([len(analytic_types)] + list(matrix_dimensions)), dtype=np.float32)
    return all_analytic_matrices


def make_interaction_matrices(num_species: int,
                              interaction_strengths: np.ndarray,
                              num_unique_interactions: int,
                              starting_iteration: int,
                              end_iteration: int):

    def idx_sampler(i, size_interaction_array, unique_interaction):
        return jnp.mod(i / jnp.power(size_interaction_array, unique_interaction), size_interaction_array).astype(int)

    unique_interaction = np.arange(num_unique_interactions)
    idx_sampler = partial(idx_sampler, size_interaction_array=np.size(
        interaction_strengths), unique_interaction=unique_interaction)

    idxs = np.arange(starting_iteration, end_iteration)[:, None]
    interaction_strength_choices = jax.vmap(idx_sampler)(idxs)
    flat_triangle = interaction_strengths[list(interaction_strength_choices)]
    interaction_matrices = jax.vmap(
        partial(make_symmetrical_matrix_from_sequence, side_length=num_species))(flat_triangle)
    return np.array(interaction_matrices), np.array(interaction_strength_choices)


def create_circuits(config: dict, interaction_matrices: np.ndarray):
    circuits = [None] * len(interaction_matrices)
    for i, interaction_matrix in enumerate(interaction_matrices):
        cfg = {"interactions_loaded": {
            'binding_rates_dissociation': interaction_matrix,
            'units': SIMULATOR_UNITS['IntaRNA']['rate']}
        }
        circuits[i] = construct_circuit_from_cfg( # type: ignore
            prev_configs=cfg, config_file=config)
    return circuits


def write_all(all_analytic_matrices, analytic_types, data_writer, out_type='npy'):
    for m, analytic in enumerate(analytic_types):
        data_writer.output(out_type=out_type, out_name=analytic,
                           data=all_analytic_matrices[m].astype(np.float32), overwrite=True,
                           write_to_top_dir=True)


def simulate(circuits: list, config: dict, data_writer):
    modeller = CircuitModeller(result_writer=data_writer, config=config)
    methods = {
        'init_circuits': {'batch': True},
        'simulate_signal_batch': {'batch': True},
    }
    if config['experiment']['debug_mode']:
        methods['write_results'] = {
            'no_visualisations': False,
            'no_numerical': False}

    return modeller.batch_circuits(
        circuits=circuits,
        methods=methods
    )


def run_batch(batch_i: int,
              batch_size: int,
              config: dict,
              data_writer,
              interaction_strengths: np.ndarray,
              num_species: int,
              num_unique_interactions: int,
              ):
    starting_iteration = int(batch_size * batch_i)
    end_iteration = int(batch_size * (batch_i + 1))

    interaction_matrices, interaction_strength_choices = make_interaction_matrices(
        num_species, interaction_strengths, num_unique_interactions, starting_iteration, end_iteration)

    circuits = create_circuits(config, interaction_matrices)
    circuits = simulate(circuits, config, data_writer)
    return circuits, interaction_strength_choices


def main(config: Optional[dict] = None, data_writer=None):

    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        'synbio_morpher', 'scripts', 'parameter_based_simulation', 'configs', 'base_config.json'))
    config = load_json_as_dict(config)
    config = prepare_config(config)

    def core_func(config):
        batch_size = int(config['batch_size'])
        interaction_strengths = create_parameter_range(
            config['parameter_based_simulation'])

        num_species = len(config['data'])
        num_unique_interactions = np.sum(
            np.triu(np.ones((num_species, num_species)))).astype(int)
        analytic_types = get_analytics_types_all()

        all_analytic_matrices = define_matrices(num_species=num_species,
                                                size_interaction_array=np.size(
                                                    interaction_strengths),
                                                num_unique_interactions=num_unique_interactions,
                                                analytic_types=analytic_types)

        # Set loop vars
        total_iterations = np.power(
            np.size(interaction_strengths), num_unique_interactions).astype(int)
        num_iterations = int(total_iterations / batch_size)

        logging.info(f'Running for a total of {total_iterations} iterations: ')
        logging.info(
            f'Each analytics matrix has a shape of {np.shape(all_analytic_matrices)[1:]}')

        save = 10
        save_every = int(num_iterations / save)
        start_time = datetime.now()
        for batch_i in range(config.get('batch_to_skip_to', 0), num_iterations):
            logging.info(
                f'\n\nBatch {batch_i} out of {num_iterations}, each of size {batch_size}. Run time so far: {datetime.now() - start_time}\n')
            circuits, all_interaction_strength_choices = run_batch(batch_i=batch_i,
                                                                   config=config,
                                                                   data_writer=data_writer,
                                                                   interaction_strengths=interaction_strengths,
                                                                   batch_size=batch_size,
                                                                   num_species=num_species,
                                                                   num_unique_interactions=num_unique_interactions)

            for circuit, interaction_strength_choices in zip(circuits, all_interaction_strength_choices):
                idxs = [slice(0, num_species)] + [[strength_idx]
                                                  for strength_idx in interaction_strength_choices]

                sig_analytics = circuit.result_collector.results['signal'].analytics
                if sig_analytics is not None:
                    for j, analytic in enumerate(get_true_names_analytics(sig_analytics)):
                        all_analytic_matrices[j][tuple(
                            idxs)] = sig_analytics[analytic][np.array([i for i, s in enumerate(circuit.model.species) if s in circuit.get_input_species()])][:, None]

            if ((batch_i > 0) and (np.mod(batch_i, save_every) == 0)) or (batch_i + 1 == num_iterations):
                write_all(all_analytic_matrices, get_true_names_analytics(
                    circuits[0].result_collector.results['signal'].analytics), data_writer)
            
            batch_time = (datetime.now() - start_time) / (batch_i + 1)
            logging.info(f'\n\nTime per batch: {batch_time}\nProjected total time: {batch_time * num_iterations}')

    experiment = Experiment(config=config, config_file=config,
                            protocols=[
                                Protocol(partial(core_func, config=config))],
                            data_writer=data_writer)
    experiment.run_experiment()
    return config, data_writer


if __name__ == "__main__":
    Fire(main)
