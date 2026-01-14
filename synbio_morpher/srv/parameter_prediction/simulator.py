
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    

import numpy as np
import diffrax as dfx
import os
import logging
from subprocess import PIPE, run
from functools import partial
from synbio_morpher.utils.misc.helper import vanilla_return, processor
from synbio_morpher.utils.modelling.physical import equilibrium_constant_reparameterisation, eqconstant_to_rates
from synbio_morpher.utils.misc.string_handling import make_time_str
from synbio_morpher.utils.data.data_format_tools.manipulate_fasta import write_fasta_file


SIMULATOR_UNITS = {
    'IntaRNA': {
        'energy': 'kcal/mol',
        'postprocessing': {
            'energy': 'J/mol',
            'energy_molecular': 'J/molecule'
        },
        'rate': r'$s^{-1}$',
        'coupled_rates': r'$s^{-1}$'
    },
}
NO_INTERACTION_EQCONSTANT = 0


class RawSimulationHandling():

    def __init__(self, config: dict) -> None:
        self.simulator_name = config.get('name', 'IntaRNA')
        self.postprocess = config.get('postprocess')
        self.sim_kwargs = config.get('simulator_kwargs', {})
        self.fixed_rate_k_a = config.get(
            'molecular_params', {}).get('association_binding_rate' + '_per_molecule')
        self.units = ''

    def get_sim_interpretation_protocol(self):

        def process_IntaRNA_sample(sample: dict):
            """ There are a variety of parameters that IntaRNA spits out. E is hybridisation energy"""
            return {
                'energies': sample.get('E', NO_INTERACTION_EQCONSTANT),
                'binding': sample.get('bpList', sample.get('subseqDB', ''))
            }

        def process_interaction(sample):
            if sample == False:  # No interactions
                sample = {}
            if type(sample) == str:
                d = process_raw_stdout(sample)
                if 'energies' not in d:
                    logging.warning(f'The sample {sample} may have been processed incorrectly. Make sure `raw_stdout` is not True in config.')
                return process_raw_stdout(d)
            return process_IntaRNA_sample(sample)
        
        if self.simulator_name == "IntaRNA":
            return process_interaction
        else:
            raise NotImplementedError(f'Simulator {self.simulator_name} has not been implemented.')

    def get_postprocessing(self, **processor_kwrgs):

        def return_both_eqconstants_and_rates(eqconstants):
            return eqconstants, eqconstant_to_rates(eqconstants, self.fixed_rate_k_a)

        if self.simulator_name == "IntaRNA":
            if self.postprocess:
                self.units = SIMULATOR_UNITS[self.simulator_name]['rate']
                processor_f = [
                    partial(
                        equilibrium_constant_reparameterisation, 
                        **processor_kwrgs),
                    return_both_eqconstants_and_rates
                ]
                return partial(
                    processor, funcs=processor_f)
            else:
                return vanilla_return
        else:
            raise NotImplementedError(f'Simulator {self.simulator_name} has not been implemented.')

    def get_simulator(self, allow_self_interaction: bool):

        if self.simulator_name == "IntaRNA":
            from synbio_morpher.srv.parameter_prediction.IntaRNA.bin.copomus.IntaRNA import IntaRNA
            self.units = SIMULATOR_UNITS[self.simulator_name]['energy']
            potential_path = check_IntaRNA_path()
            if potential_path is not None:
                import sys
                sys.path.append(potential_path)
            return partial(simulate_IntaRNA,
                           allow_self_interaction=allow_self_interaction,
                           sim_kwargs=self.sim_kwargs,
                           simulator=IntaRNA())

        elif self.simulator_name == "CopomuS":
            # from synbio_morpher.utils.parameter_prediction.IntaRNA.bin.CopomuS import CopomuS
            raise NotImplementedError(f'CopomuS has not yet been implemented')

        else:
            raise NotImplementedError(f'Simulator {self.simulator_name} has not been implemented.')


def check_IntaRNA_path():

    p = run('which IntaRNA', shell=True, stdout=PIPE,
            stderr=PIPE, universal_newlines=True)
    p2 = run('IntaRNA --version', shell=True, stdout=PIPE,
             stderr=PIPE, universal_newlines=True)
    if p.returncode == 0:
        if p2.returncode == 1:
            return p.stdout
    else:
        logging.warning(f'Could not detect IntaRNA on system: {p}')
        
        
def make_piecewise_stepcontrol(t0, t1, dt0, dt1, split: int = 3):
    tdiff = (t1 - t0) / split
    dts = np.interp(np.arange(split), [0, split-1], [dt0, dt1])

    i = 0
    ts = np.arange(
            t0 + tdiff * i, 
            t0 + tdiff * i + tdiff, 
            dts[i])
    for i in range(i+1, split):
        nx = np.arange(
            t0 + tdiff * i, 
            t0 + tdiff * i + tdiff, 
            dts[i])
        ts = np.concatenate([ts, nx])
    ts[0] = t0
    ts[-1] = t1
    return dfx.StepTo(ts)


def process_raw_stdout(stdout):
    """ Process the raw output of IntaRNA when multithreading. """
    d = {}
    header = stdout.split('\n')[0].split(';')
    for t in stdout.split('\n')[1:-1]:
        d.setdefault(t.split(';')[0], {})
        d[t.split(';')[0]][t.split(';')[1]] = {n: t.split(';')[i] for i, n in enumerate(header)}
    return d


def write_temp_fastas(input: dict):
    temp_fn = 'temp_circuit'
    while os.path.exists(temp_fn):
        temp_fn = temp_fn + '_' + make_time_str()
        if len(temp_fn) > 100:
            raise ValueError('The temporary directory for writing fastas to simulate interactions could not be created.')
        
    write_fasta_file(out_path=temp_fn, data=input, byseq=True)
    return temp_fn


def simulate_IntaRNA(input, compute_by_filename: bool, allow_self_interaction: bool, sim_kwargs: dict, simulator, filename=None):
    if compute_by_filename:
        f = simulate_IntaRNA_fn
    elif sim_kwargs['threads'] > 0:
        f = partial(simulate_IntaRNA_fn, remove_file=True)
        filename = write_temp_fastas(dict(zip(list(map(lambda s: s.name, input.keys())), input.values())))
        input = (filename, input)
    else:
        f = simulate_IntaRNA_data
    return f(input, allow_self_interaction=allow_self_interaction, sim_kwargs=sim_kwargs, simulator=simulator) # type: ignore


def simulate_IntaRNA_data(batch: dict, allow_self_interaction: bool, sim_kwargs: dict, simulator):
    """ Possible outputs of IntaRNA can be found in their README https://github.com/BackofenLab/IntaRNA 
    Simply add any parameter column id's (E, ED1, seq1, ...) into the simulation arguments dict using the 
    `outcsvcols` variable. """

    if batch is not None:
        data = {}
        for i, (label_i, sample_i) in enumerate(batch.items()):
            current_pair = {}
            for j, (label_j, sample_j) in enumerate(batch.items()):
                if not allow_self_interaction and i == j:
                    continue
                if i > j:  # Skip symmetrical
                    current_pair[label_j] = data[label_j][label_i]
                else:
                    sim_kwargs["query"] = sample_i
                    sim_kwargs["target"] = sample_j
                    current_pair[label_j] = simulator.run(**sim_kwargs)
                    if type(current_pair[label_j]) == dict and ('id1' in current_pair[label_j].keys()) and ('id2' in current_pair[label_j].keys()):
                        current_pair[label_j]['id1'] = label_j if current_pair[label_j]['id1'] == 'target' else label_i
                        current_pair[label_j]['id2'] = label_j if current_pair[label_j]['id1'] == 'target' else label_i
            data[label_i] = current_pair
    else:
        data = simulator.run(**sim_kwargs)
    return data


def simulate_IntaRNA_fn(inputs: tuple, allow_self_interaction: bool, sim_kwargs: dict, simulator, remove_file: bool = False):
    """ Use the FASTA filename to compute the interactions between all RNAs.
    If threads = 0, interactions will be computed in parallel """
    filename, input = inputs
    dinit = dict(zip(input.keys(), [False] * len(input)))
    data = dict(zip(input.keys(), [dinit] * len(input.keys())))
    species_str = dict(zip(list(map(lambda s: s.name, input.keys())), input.keys()))

    sim_kwargs["query"] = filename
    sim_kwargs["target"] = filename
    output = simulator.run(**sim_kwargs)
    if output:
        if type(output) == str:
            output = list(process_raw_stdout(output).values())
        if type(output) == dict:
            output = [output]
        for s in output:
            if not allow_self_interaction and s['id1'] == s['id2']:
                continue
            data[species_str[s['id1']]][species_str[s['id2']]] = s # type: ignore
    if remove_file:
        os.remove(filename)
    return data
