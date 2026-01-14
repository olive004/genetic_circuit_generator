
# Copyright (c) 2024, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import os
from fire import Fire
import numpy as np
from typing import List

from synbio_morpher.utils.common.setup import construct_circuit_from_cfg, prepare_config
from synbio_morpher.utils.results.experiments import Experiment, Protocol
from synbio_morpher.srv.io.manage.script_manager import script_preamble
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.data.fake_data_generation.energies import generate_energies
from synbio_morpher.utils.circuit.agnostic_circuits.circuit_manager import CircuitModeller
from synbio_morpher.srv.parameter_prediction.simulator import RawSimulationHandling


def preprocess_energies(energies: np.ndarray, quantities: np.ndarray, sim_args: dict, stype: str) -> List[dict]:
    """ Convert energies into equilibrium constants and binding rates.
    `energies` is [n_circuits, n_species, n_species] """
    sample_postproc = RawSimulationHandling(sim_args).get_postprocessing(
        initial=quantities)
    eqconstants, (a_rates, d_rates) = sample_postproc(energies)
    return [{
        'name': f'circuit_{i}',
        'data': {
            f'{stype}_{ii}': '' for ii in range(len(quantities))
        },
        'interactions_loaded': {
            'eqconstants': eq,
            'binding_rates_association': ba,
            'binding_rates_dissociation': bd,
            'energies': en,
        }} for i, (eq, ba, bd, en) in enumerate(zip(eqconstants, a_rates, d_rates, energies))]


# def rm_dupes(data_writer):
#     dirs_to_rem = ['eqconstants', 'binding_rates_association',
#                    'binding_rates_dissociation', 'energies', 'binding_sites']
#     for dir_to_rem in dirs_to_rem:
#         dn = os.path.join(data_writer.ensemble_write_dir, dir_to_rem)
#         if os.path.exists(dn):
#             for root, dirs, files in os.walk(dn, topdown=False):
#                 for name in files:
#                     os.remove(os.path.join(root, name))
#                 for name in dirs:
#                     os.rmdir(os.path.join(root, name))
#             os.rmdir(dn)


def main(config=None, data_writer=None):
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "synbio_morpher", "scripts", "simulate_by_interaction", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)
    config_file = prepare_config(config_file)
    exp_configs = config_file["circuit_generation"]
    sim_args = config['interaction_simulator']

    protocols = [
        Protocol(
            partial(generate_energies,
                    n_circuits=exp_configs.get("repetitions", 1),
                    n_species=exp_configs.get("species_count", 3), len_seq=exp_configs["sequence_length"],
                    p_null=exp_configs.get("perc_non_interacting", 0.3),
                    symmetrical=True if config_file.get(
                        "system_type", "RNA") == 'RNA' else False,
                    type_energies=config_file.get("system_type", "RNA"),
                    seed=exp_configs.get("seed", 0)),
            req_output=True,
            name="generating_energies"
        ),
        Protocol(
            partial(
                preprocess_energies,
                sim_args=sim_args,
                quantities=config_file.get('molecular_params', {}).get(
                    "starting_copynumbers", 100) * np.ones(exp_configs.get("species_count", 3)),
                stype=config_file.get("system_type", "RNA")
            ),
            req_input=True,
            req_output=True,
            name="preprocess_energies"
        ),
        [
            Protocol(
                partial(construct_circuit_from_cfg,
                        config_file=config_file),
                req_input=True,
                req_output=True,
                name="making_circuit"
            ),
            # Protocol(
            #     CircuitModeller(
            #         result_writer=data_writer, config=config_file).write_interactions,
            #     req_input=True,
            #     req_output=True,
            #     name="compute_interaction_strengths"
            # )
        ],
        Protocol(partial(CircuitModeller(result_writer=data_writer, config=config_file).batch_circuits,
                         write_to_subsystem=True, batch_size=config_file['simulation'].get('batch_size', 100),
                         methods={
            "compute_interactions": {},
            "write_interactions": {},
            "init_circuits": {'batch': True},
            "simulate_signal_batch": {'ref_circuit': None,
                                      'batch': True},
            "write_results": {'no_visualisations': config_file['experiment'].get('no_visualisations', True),
                              'no_numerical': config_file['experiment'].get('no_numerical', False)}
        }),
            req_input=True,
            name="simulate"
        )
        # Protocol(partial(rm_dupes, data_writer=data_writer),
        #          req_input=False, name="remove_duplicates")
    ]

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()

    return config, data_writer


if __name__ == "__main__":
    Fire(main)
