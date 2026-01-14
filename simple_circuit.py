import numpy as np
import pandas as pd
import importlib
import os
from synbio_morpher.utils.common.setup import construct_circuit_from_cfg, prepare_config
from synbio_morpher.srv.io.manage.script_manager import script_preamble
from synbio_morpher.srv.io.manage.sys_interface import PACKAGE_NAME
from synbio_morpher.utils.evolution.evolver import Evolver
from synbio_morpher.utils.circuit.agnostic_circuits.circuit_manager import CircuitModeller


def five_circuits():

    config = {
        "interaction_simulator": {
            "name": "IntaRNA",
            "postprocess": True
        },
        "experiment": {
            "purpose": "tests",
            "debug_mode": True
        },
        "molecular_params": "./synbio_morpher/utils/common/configs/RNA_circuit/molecular_params.json",
        "circuit_generation": {
            "repetitions": 2000,
            "species_count": 3,
            "sequence_length": 20,
            "generator_protocol": "template_mutate",
            "proportion_to_mutate": 0.5
        },
        "mutations_args": {
            "algorithm": "random",
            "mutation_counts": 1,
            "mutation_nums_within_sequence": [1],
            "mutation_nums_per_position": 1,
            "concurrent_species_to_mutate": "single_species_at_a_time"
        },
        "filters": {
            "min_num_interacting": None,
            "max_self_interacting": None,
            "max_total": None
        },
        "signal": {
            "inputs": ["RNA0"],
            "outputs": ["RNA1"],
            "function_name": "step_function",
            "function_kwargs": {
                "impulse_center": 400,
                "impulse_halfwidth": 5,
                "target": 10
            }
        },
        "simulation": {
            "dt0": 0.1,
            "t0": 0,
            "t1": 1200,
            "solver": "diffrax",
            "use_batch_mutations": True,
            "batch_size": 100,
            "max_circuits": 1000,
            "device": "cpu"
        },
        "system_type": "RNA"
    }
    config, data_writer = script_preamble(config=config, data_writer=None)
    config = prepare_config(config_file=config)

    default_interaction = 0.0015
    species_num = 9
    
    cfg_dir = os.path.join(PACKAGE_NAME, 'utils', 'common', 'testing', 'configs')
    cfg_dir = importlib.import_module(cfg_dir.replace(os.sep, '.')).__path__[0]

    paths = [
        # toy_mRNA_circuit_1890
        os.path.join(cfg_dir, 'circuits', '0_weak.fasta'),
        # toy_mRNA_circuit_1916
        os.path.join(cfg_dir, 'circuits', '1_med_weak.fasta'),
        # toy_mRNA_circuit_1598
        os.path.join(cfg_dir, 'circuits', '2_medium.fasta'),
        # toy_mRNA_circuit_1196
        os.path.join(cfg_dir, 'circuits', '3_med_strong.fasta'),
        # toy_mRNA_circuit_1491
        os.path.join(cfg_dir, 'circuits', '4_strong.fasta')
    ]
    
    interaction_paths = []
    for inter in ['binding_rates_dissociation', 'eqconstants']:
        interaction_paths.append([
            # toy_mRNA_circuit_1890
            os.path.join(cfg_dir, inter, f'0_weak_{inter}.csv'),
            # toy_mRNA_circuit_1916
            os.path.join(cfg_dir, inter, f'1_med_weak_{inter}.csv'),
            # toy_mRNA_circuit_1598
            os.path.join(cfg_dir, inter, f'2_medium_{inter}.csv'),
            # toy_mRNA_circuit_1196
            os.path.join(cfg_dir, inter,
                         f'3_med_strong_{inter}.csv'),
            # toy_mRNA_circuit_1491
            os.path.join(cfg_dir, inter, f'4_strong_{inter}.csv')
        ])

    # interactions = np.expand_dims(np.expand_dims(np.arange(
    #     len(paths)), axis=1), axis=2) * np.ones((len(paths), species_num, species_num)) * default_interaction
    interactions_cfg = [
        {'binding_rates_association': config['molecular_params']
            ['creation_rate'], 'binding_rates_dissociation': bp, 'eqconstants': ep}
        for bp, ep in zip(interaction_paths[0], interaction_paths[1])]

    return [construct_circuit_from_cfg(
        {'data_path': p, 'interactions': i}, config) for p, i in zip(paths, interactions_cfg)], config, data_writer


def mutate(circuits, config, data_writer):

    for c in circuits:
        c = Evolver(data_writer=data_writer,
                    sequence_type=config.get('system_type')).mutate(
            c,
            write_to_subsystem=True,
            algorithm=config.get('mutations_args', {}).get('algorithm', 'random'))
    return circuits, config, data_writer


def simulate(circuits, config, data_writer):

    CircuitModeller(result_writer=data_writer, config=config).batch_circuits(
        circuits=circuits,
        write_to_subsystem=True,
        batch_size=config['simulation'].get('batch_size', 100),
        methods={
            "compute_interactions": {},
            "init_circuits": {'batch': True},
            "simulate_signal_batch": {'ref_circuit': None,
                                      'batch': True},
            "write_results": {'no_visualisations': False,
                              'no_numerical': False}
        })



circuits, config, data_writer = five_circuits()
mutate(circuits, config, data_writer)
simulate(circuits, config, data_writer)
