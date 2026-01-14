

from bioreaction.simulation.manager import simulate_steady_states
from functools import partial
from typing import Optional, Tuple, List, Union
from fire import Fire
from datetime import datetime
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import logging


from synbio_morpher.utils.circuit.agnostic_circuits.circuit_manager import CircuitModeller
from synbio_morpher.utils.common.setup import construct_circuit_from_cfg, prepare_config
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.data.data_format_tools.manipulate_fasta import load_seq_from_FASTA
from synbio_morpher.utils.evolution.evolver import Evolver, apply_mutation_to_sequence
from synbio_morpher.utils.evolution.mutation import get_mutation_type_mapping, reverse_mut_mapping
from synbio_morpher.utils.misc.type_handling import flatten_listlike
from synbio_morpher.utils.results.analytics.naming import get_true_interaction_cols
from synbio_morpher.utils.results.analytics.timeseries import log_distance, sp_prod, vec_distance
from synbio_morpher.utils.results.experiments import Experiment, Protocol
from synbio_morpher.utils.results.writer import DataWriter
from synbio_morpher.srv.io.loaders.circuit_loader import load_circuit
from synbio_morpher.srv.io.manage.script_manager import script_preamble


# Init

def pick_circuits(config: dict, data: pd.DataFrame, init_sample_size: int = 1000) -> pd.DataFrame:

    percentile = 0.9
    data = data.iloc[np.random.choice(
        np.arange(len(data)), size=init_sample_size, replace=False)]

    sensitivity_range = data['sensitivity'] > (data['sensitivity'].max() *
                                                             percentile)
    starting_circ_rows = data[sensitivity_range].sort_values(
        by=['sensitivity', 'precision'], ascending=False)

    logging.info(
        f'Picking circuits that have a sensitivity in the {percentile * 100}th percentile of at least {data["sensitivity"].max() * percentile}')
    return starting_circ_rows


def tweak_cfg(config_og: dict, config: dict) -> dict:
    config_og['experiment']['purpose'] = config['experiment']['purpose']
    config_og['simulation']['device'] = config['simulation']['device']
    config_og['mutations_args'] = {
        'algorithm': 'random',
        'mutation_counts': 2,
        'mutation_nums_within_sequence': [1],
        'mutation_nums_per_position': 1,
        'concurrent_species_to_mutate': 'single_species_at_a_time',
        'seed': 0
    }
    config_og['simulation']['dt0'] = config_og['simulation'].get('dt', 0.1)
    config_og['simulation']['threshold_steady_states'] = 0.1
    config_og['experiment']['no_numerical'] = False
    config_og['experiment']['no_visualisations'] = True
    for k, v in config.items():
        if k in config_og and type(config_og[k]) == dict:
            config_og[k].update(v)
        else:
            config_og[k] = v
    return config_og


def plot_starting_circuits(starting_circ_rows, data, filt, data_writer, save: bool = True):
    data['Starting circuit'] = (data['circuit_name'].isin(starting_circ_rows['circuit_name'])) & \
        (data['mutation_name'].isin(starting_circ_rows['mutation_name'])) & filt
    sns.scatterplot(
        data.sort_values(by='Starting circuit'), x='sensitivity', y='precision',
        hue='Starting circuit', alpha=((data.sort_values(by='Starting circuit')['Starting circuit'] + 0.1)/1.1)
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sensitivity')
    plt.ylabel('Precision')
    fig_path = os.path.join(data_writer.top_write_dir, 'chosen_circuits.png')
    plt.savefig(fig_path)


def make_distance_func(data):
    sp_min = data[(
        data['sensitivity'] <= 1/data['precision']) | (
            data['precision'] <= 1/data['sensitivity'])][['sensitivity', 'precision']].min().to_numpy()
    sp_max = data[(
        data['sensitivity'] <= 1/data['precision']) | (
            data['precision'] <= 1/data['sensitivity'])][['sensitivity', 'precision']].max().to_numpy()
    sp_left = np.array([sp_min[0], sp_max[1]])
    sp_right = np.array([sp_max[0], sp_min[1]])
    d = np.array([sp_left, sp_right]).T
    return partial(vec_distance, d=d)


# Simulation functions

def mutate(circuits: list, evolver: Evolver, algorithm: str = 'random'):
    for c in circuits:
        c = evolver.mutate(
            c, write_to_subsystem=True, algorithm=algorithm)
    return circuits


def simulate(circuits: list, modeller: CircuitModeller, config: dict) -> list:
    circuits = modeller.batch_circuits(
        circuits=circuits,
        write_to_subsystem=True,
        batch_size=config['simulation'].get('batch_size', 100),
        methods={
            "compute_interactions": {},
            "init_circuits": {'batch': True},
            "simulate_signal_batch": {'ref_circuit': None,
                                      'batch': config['simulation']['use_batch_mutations']},
            "write_results": {'no_visualisations': config['experiment']['no_visualisations'],
                              'no_numerical': config['experiment']['no_numerical']}
        }
    )
    return circuits


# Helper functions

def load_all_analytics(circuit_dirs):
    """ Top (starting) directory should be the starting circuit for each round """
    fbn = 'report_signal.json'
    circuits = {}
    for circuit_dir in circuit_dirs:
        if 'mutations' in os.listdir(circuit_dir):
            circuits[os.path.basename(circuit_dir)] = load_all_analytics(
                [os.path.join(circuit_dir, sc) for sc in os.listdir(circuit_dir)])
            circuits[os.path.basename(circuit_dir)]['ref_circuit'] = load_json_as_dict(
                os.path.join(circuit_dir, fbn))
        else:
            circuits[os.path.basename(circuit_dir)] = load_json_as_dict(
                os.path.join(circuit_dir, fbn))
    return circuits


def make_next_name(name: str):
    special_char = '_N_'
    return name.split(special_char)[0] + special_char + ''.join(str(datetime.now()).replace('.', '').replace(':', '').split(' ')[1:])


# Choose next
def choose_next(batch: list, data_writer, distance_func, choose_max: int = 4, target_species: List[str] = ['RNA_1', 'RNA_2'], use_diversity: bool = False):

    def make_data(batch, batch_analytics, target_species: List[str]):
        d = pd.DataFrame(
            data=np.concatenate(
                [
                    np.asarray([c.name for c in batch])[:, None],
                    np.asarray([c.subname for c in batch])[:, None]
                ], axis=1
            ),
            columns=['Name', 'Subname']
        )
        d['Circuit Obj'] = batch
        species_names = [s.name for s in batch[0].model.species]
        t_idxs = {s: species_names.index(
            s) for s in species_names if s in target_species}
        for t in target_species:
            t_idx = t_idxs[t]
            d[f'Sensitivity species-{t}'] = np.asarray(
                [b['sensitivity'][t_idx] for b in batch_analytics])
            d[f'Precision species-{t}'] = np.asarray(
                [b['precision'][t_idx] for b in batch_analytics])
            d[f'Overshoot species-{t}'] = np.asarray(
                [b['overshoot'][t_idx] for b in batch_analytics])

            rs = d[d['Subname'] == 'ref_circuit']
            d[f'Parent Sensitivity species-{t}'] = jax.tree_util.tree_map(
                lambda n: rs[rs['Name'] == n][f'Sensitivity species-{t}'].iloc[0], d['Name'].to_list())
            d[f'Parent Precision species-{t}'] = jax.tree_util.tree_map(
                lambda n: rs[rs['Name'] == n][f'Precision species-{t}'].iloc[0], d['Name'].to_list())

            # d[f'dS species-{t}'] = np.asarray([b['sensitivity_diff_to_base_circuit'][t_idx] for b in batch_analytics])
            # d[f'dP species-{t}'] = np.asarray([b['precision_diff_to_base_circuit'][t_idx] for b in batch_analytics])
            d[f'dS species-{t}'] = d[f'Sensitivity species-{t}'] - \
                d[f'Parent Sensitivity species-{t}']
            d[f'dP species-{t}'] = d[f'Precision species-{t}'] - \
                d[f'Parent Precision species-{t}']

            # d[f'Diag Distance species-{t}'] = distance_func(s=d[f'Sensitivity species-{t}'].to_numpy(), p=d[f'Precision species-{t}'].to_numpy())
            d[f'SP Prod species-{t}'] = sp_prod(s=d[f'Sensitivity species-{t}'].to_numpy(), p=d[f'Precision species-{t}'].to_numpy(),
                                                # (d[f'Precision species-{t}'] / d[f'Sensitivity species-{t}']).max(),
                                                sp_factor=1,
                                                s_weight=0)  # np.log(d[f'Precision species-{t}']) / d[f'Sensitivity species-{t}'])
            d[f'Log Distance species-{t}'] = np.array(log_distance(
                s=d[f'Sensitivity species-{t}'].to_numpy(), p=d[f'Precision species-{t}'].to_numpy()))
            # d[f'SP and distance species-{t}'] = np.log( np.power(d[f'Log Distance species-{t}'], dist_weight) * np.log(d[f'SP Prod species-{t}']))
            d[f'SP and distance species-{t}'] = d[f'Sensitivity species-{t}'] * \
                d[f'Log Distance species-{t}']

        return d

    def select_next(data_1, choose_max, t, use_diversity: bool):
        # filt = (data_1[f'dS species-{t}'] >= 0) & (data_1[f'dP species-{t}'] >= 0) & (
        #     data_1[f'Sensitivity species-{t}'] >= data_1[data_1['Subname'] == 'ref_circuit'][f'Sensitivity species-{t}'].min()) & (
        #         data_1[f'Precision species-{t}'] >= data_1[data_1['Subname'] == 'ref_circuit'][f'Precision species-{t}'].min())

        data_1['Diversity selection'] = False
        circuits_chosen = data_1.sort_values(
            by=[f'SP and distance species-{t}', f'Log Distance species-{t}', f'SP Prod species-{t}', 'Name', 'Subname'], ascending=False)['Circuit Obj'].iloc[:choose_max].to_list()
        prev_circuits = data_1[data_1['Subname'] == 'ref_circuit']
        keep_n = int(0.7 * choose_max)
        if use_diversity and all([c in prev_circuits for c in circuits_chosen]) and (len(data_1) >= keep_n):
            _, circuits_chosen = select_next(
                data_1[data_1['Circuit Obj'].isin(prev_circuits[:keep_n])], choose_max, t, use_diversity)
            data_1['Diversity selection'] = data_1['Circuit Obj'].isin(
                circuits_chosen)

        data_1['Next selected'] = data_1['Circuit Obj'].isin(circuits_chosen)
        return data_1, circuits_chosen

    def get_batch_analytics(batch, data_writer):
        batch_analytics = []
        for c in batch:
            if c.subname == 'ref_circuit':
                batch_analytics.append(
                    load_json_as_dict(os.path.join(data_writer.top_write_dir, c.name, 'report_signal.json')))
            else:
                batch_analytics.append(
                    load_json_as_dict(os.path.join(
                        data_writer.top_write_dir, c.name, 'mutations', c.subname, 'report_signal.json'))
                )
        batch_analytics = jax.tree_util.tree_map(
            lambda x: np.float64(x), batch_analytics)
        return batch_analytics

    batch_analytics = get_batch_analytics(batch, data_writer)
    data_1 = make_data(batch, batch_analytics, target_species)

    t = target_species[0]
    # circuits_chosen = data_1[(data_1[f'dS species-{t}'] >= 0) & (data_1[f'dP species-{t}'] >= 0)].sort_values(by=[f'Sensitivity species-{t}', f'Precision species-{t}'], ascending=False)['Circuit Obj'].iloc[:choose_max].to_list()
    data_1, circuits_chosen = select_next(data_1, choose_max, t, use_diversity)
    return circuits_chosen, data_1


# Process mutations between runs

def get_mutated_sequences(path, circ_row, mutation_type_mapping) -> Union[str, dict]:

    if not os.path.isfile(path):
        path = os.path.join('..', path)
        assert os.path.isfile(path), f'Path {path} is not valid.'

    if circ_row['mutation_name'] == 'ref_circuit':
        return path

    sequences = load_seq_from_FASTA(path, as_type='dict')
    if type(sequences) == dict:
        mutated_species = circ_row['mutation_name'][:5]
        mutation_types = jax.tree_util.tree_map(
            lambda x: mutation_type_mapping[x], circ_row['mutation_type'])
        mutated_sequence = apply_mutation_to_sequence(
            sequences[mutated_species], circ_row['mutation_positions'], mutation_types)

        sequences[mutated_species] = mutated_sequence
        return sequences
    else:
        raise ValueError(
            f'Sequences should be a dictionary when loaded from FASTA instead of {type(sequences)}: {sequences}')


def process_for_next_run(circuits: list, data_writer: DataWriter):

    for i, c in enumerate(circuits):
        circuits[i].name = make_next_name(c.name)
        # sequences = {s.name: s.physical_data for s in c.model.species if s.physical_data}
        sequences = load_seq_from_FASTA(c.data.source, as_type='dict')
        circuits[i].data.source = write_mutated_circuit(
            name=circuits[i].name, subname='ref_circuit', sequences=sequences, data_writer=data_writer)
    return circuits


def write_mutated_circuit(
        name: str, subname: str, sequences, data_writer: DataWriter):

    data_writer.subdivide_writing(name)
    if subname != 'ref_circuit':
        data_writer.subdivide_writing(subname, safe_dir_change=False)

    new_path = data_writer.output(
        out_name=name,
        out_type='fasta', return_path=True,
        data=sequences, byseq=True
    )

    data_writer.unsubdivide()
    return new_path


def make_starting_circuits(starting_circuits: pd.DataFrame, config: dict, data_writer):
    d = {v: v for v in jax.tree_util.tree_flatten(
        get_mutation_type_mapping('RNA'))[0]}
    mutation_type_mapping_rev = jax.tree_util.tree_map(
        lambda x: reverse_mut_mapping(x), d)

    circuits = []
    for i, circ_row in starting_circuits.iterrows():
        curr_config = config
        sequences = get_mutated_sequences(
            circ_row['path_to_template_circuit'], circ_row, mutation_type_mapping_rev)
        path = write_mutated_circuit(
            name=circ_row['circuit_name'],
            subname=circ_row['mutation_name'],
            sequences=sequences,
            data_writer=data_writer)
        curr_config['data_path'] = path
        curr_config['interactions_loaded'] = None
        curr_config['interactions'] = None
        circuit = construct_circuit_from_cfg(
            prev_configs=None, config_file=curr_config)
        circuit.name = make_next_name(circ_row['circuit_name'])
        circuits.append(circuit)

    return circuits


# Start loop

def loop(config, data_writer, modeller, evolver, starting_circ_rows, distance_func):
    target_species = ['RNA_1', 'RNA_2']
    choose_max = config['choose_max']
    total_steps = config['total_steps']

    starting = make_starting_circuits(
        starting_circ_rows.iloc[:choose_max], config, data_writer)
    starting = simulate(starting, modeller, config)

    summary = {}
    summary_batch = {}
    summary_datas = {}
    summary[0] = starting
    for step in range(total_steps):

        print(f'\n\nStarting batch {step+1} out of {total_steps}\n\n')

        batch = mutate(starting, evolver,
                       algorithm=config['mutations_args']['algorithm'])
        batch = simulate(batch, modeller, config)
        expanded_batchs = []
        for b in batch:
            config['data_path'] = b.data.source
            expanded_batchs.append(load_circuit(
                os.path.join(data_writer.top_write_dir, b.name),
                name=b.name, config=config, load_mutations_as_circuits=True))
        expanded_batchs = flatten_listlike(expanded_batchs, safe=True)
        starting, summary_data = choose_next(batch=expanded_batchs, data_writer=data_writer, distance_func=distance_func,
                                             choose_max=choose_max, target_species=target_species, use_diversity=config.get('use_diversity', False))
        starting = process_for_next_run(starting, data_writer=data_writer)

        summary[step+1] = starting
        summary_batch[step] = expanded_batchs
        summary_datas[step] = summary_data

        for i in range(len(starting)):
            starting[i].subname = 'ref_circuit'

    return summary_datas


# Visualise circuit trajectory

def visualise_step_plot(summary_datas: pd.DataFrame, data_writer, species: str):
    n_rows = int(np.ceil(np.sqrt(len(summary_datas))))
    n_cols = int(np.ceil(np.sqrt(len(summary_datas))))
    plt.figure(figsize=(7 * n_rows, 7 * n_cols))
    for i, (step, sdata) in enumerate(summary_datas.items()):
        ax = plt.subplot(n_rows, n_cols, i+1)
        sdata = sdata.sort_values(['Next selected']) # type: ignore
        sns.scatterplot(sdata, x=f'Sensitivity species-{species}', y=f'Precision species-{species}', hue='Next selected', alpha=0.1)
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Step {step}')

    fig_path = os.path.join(data_writer.top_write_dir,
                            f'step_change_{species}.svg')
    plt.savefig(fig_path)


def main(config=None, data_writer=None):

    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "synbio_morpher", "scripts", "mc_evolution", "configs", "base_config.json"))

    np.random.seed(config.get('seed', 0))

    fn = config['info_path']
    data = pd.read_csv(fn)
    data['mutation_type'] = data['mutation_type'].str.strip(
        '[]').str.split(',').apply(lambda x: [int(xx) for xx in x if xx])
    data['mutation_positions'] = data['mutation_positions'].str.strip(
        '[]').str.split(',').apply(lambda x: [int(xx) for xx in x if xx])

    config_og = load_json_as_dict(os.path.join(fn.split('summarise')[
        0], 'mutation_effect_on_interactions_signal', 'experiment.json'))
    config_og = config_og['config_filepath']
    config_og = prepare_config(config_file=config_og)
    config_og = tweak_cfg(config_og, config)

    modeller = CircuitModeller(data_writer, config_og)
    evolver = Evolver(data_writer, mutation_type='random',
                      sequence_type='RNA', seed=0)

    signal_species = config_og['signal']['inputs']
    filt = (
        (data[get_true_interaction_cols(data, 'energies')].sum(axis=1) != 0) &
        (data['sample_name'].isin(signal_species) != True) &
        (data['overshoot'] > 0)
    )

    starting_circ_rows = pick_circuits(
        config_og, data[filt], init_sample_size=config['init_sample_size'])

    plot_starting_circuits(starting_circ_rows, data, filt, data_writer)

    # summary_datas = loop(config_og, data_writer, modeller, evolver, starting_circ_rows)

    def write(summary_datas, data_writer):
        data_writer.subdivide_writing('summary_datas')
        for step, sdata in summary_datas.items():
            data_writer.output(out_type='csv', out_name='sdata_' +
                               str(step), write_master=False, data=sdata)
        data_writer.unsubdivide()

    # visualise_step_plot(summary_datas, data_writer, species='RNA_1')
    # visualise_step_plot(summary_datas, data_writer, species='RNA_2')

    protocols = [
        Protocol(
            partial(loop, config=config_og, data_writer=data_writer, modeller=modeller, evolver=evolver,
                    starting_circ_rows=starting_circ_rows, distance_func=make_distance_func(data)),
            req_output=True
        ),
        Protocol(
            partial(write, data_writer=data_writer),
            req_input=True
        ),
        Protocol(
            partial(visualise_step_plot,
                    data_writer=data_writer, species='RNA_1'),
            req_input=True
        ),
        Protocol(
            partial(visualise_step_plot,
                    data_writer=data_writer, species='RNA_2'),
            req_input=True
        )
    ]

    experiment = Experiment(config=config, config_file=config, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()

    return config, data_writer


if __name__ == "__main__":
    Fire(main)
