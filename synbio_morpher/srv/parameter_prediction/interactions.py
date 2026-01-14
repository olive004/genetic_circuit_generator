
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import numpy as np
import os
import pandas as pd
from typing import Tuple, List, Union, Optional

from synbio_morpher.srv.parameter_prediction.simulator import RawSimulationHandling
from synbio_morpher.srv.io.loaders.experiment_loading import INTERACTION_FILE_ADDONS, load_param, load_units
from synbio_morpher.srv.io.loaders.misc import load_csv
from synbio_morpher.srv.io.manage.sys_interface import PACKAGE_DIR
from synbio_morpher.utils.data.data_format_tools.common import determine_file_format
from synbio_morpher.utils.misc.string_handling import string_to_tuple_list


INTERACTION_TYPES = sorted(INTERACTION_FILE_ADDONS.keys())
INTERACTION_FIELDS_TO_WRITE = [
    'binding_sites'
] + INTERACTION_TYPES


class MolecularInteractions():

    precision = np.float32

    def __init__(self,
                 binding_rates_association: Union[np.ndarray, float],
                 binding_rates_dissociation: Union[np.ndarray, float],
                 energies: Union[np.ndarray, float],
                 eqconstants: Union[np.ndarray, float],
                 binding_sites: Union[np.ndarray, float],
                 units=None,
                 raw=None) -> None:
        self.binding_rates_association = binding_rates_association
        self.binding_rates_dissociation = binding_rates_dissociation
        self.energies = energies
        self.eqconstants = eqconstants
        self.raw = raw
        self.set_precision()

        self.binding_sites = binding_sites
        self.units = units

    def set_precision(self):
        for attr, v in self.__dict__.items():
            if v is None or (attr == 'raw'):
                continue
            self.__setattr__(attr, self.precision(v))


class InteractionMatrix():

    def __init__(self,  # config_args=None,
                 matrix_paths: Union[dict, None] = None,
                 experiment_dir: Union[None, str] = None,
                 num_nodes: Optional[int] = None,
                 units: str = '',
                 experiment_config: Union[None, dict] = None,
                 interactions_loaded: Union[None, dict] = None,
                 allow_empty=False
                 ):

        self.name = None
        self.sample_names = None
        self.units = units
        self.experiment_dir = experiment_dir
        self.experiment_config = experiment_config

        # if num_nodes is None:
        #     num_nodes = 1
        nans = np.nan  # np.zeros((num_nodes, num_nodes)) * np.nan
        self.interactions = MolecularInteractions(
            binding_rates_association=nans,
            binding_rates_dissociation=nans,
            energies=nans,
            eqconstants=nans,
            binding_sites=nans,
            units='test'
        )
        set_assoc = False

        if interactions_loaded is not None:
            for matrix_type, loaded_matrix in interactions_loaded.items():
                self.interactions.__setattr__(matrix_type, loaded_matrix)
            if (interactions_loaded.get('binding_rates_dissociation') is not None) and (
                    interactions_loaded.get('binding_rates_association') is None):
                set_assoc = True

        if matrix_paths is not None:
            for matrix_type, matrix_path in matrix_paths.items():
                if isinstance(matrix_path, str):
                    loaded_matrix, self.units, self.sample_names = self.load(
                        matrix_path)
                    self.interactions.__setattr__(matrix_type, loaded_matrix)
                    self.interactions.units = self.units
                elif isinstance(matrix_path, np.ndarray):
                    self.interactions.__setattr__(matrix_type, matrix_path)
                    self.interactions.units = self.units
            if 'binding_rates_association' in matrix_paths:
                self.interactions.binding_rates_association = matrix_paths['binding_rates_association'] * np.ones_like(
                    self.interactions.binding_rates_dissociation)
            elif not allow_empty:
                set_assoc = True
        if set_assoc and ((self.interactions.binding_rates_association is None) or np.isnan(self.interactions.binding_rates_association).all()):
            assert experiment_config is not None, f'Please either provide the parameter for `association_binding_rate` as `binding_rates_association` in the `interactions` field in the config, or provide the entire config.'
            self.interactions.binding_rates_association = load_param(
                filepath=None, param='association_binding_rate', experiment_config=experiment_config
            ) * np.ones_like(self.interactions.binding_rates_dissociation)
            
        self.post_init()
            
    def post_init(self):
        vals = [len(v) for v in self.interactions.__dict__.values() if type(v) == np.array]
        if vals:
            max_len = np.max(vals)
            for k, v in self.interactions.__dict__.items():
                if (type(v) != np.array) and (type(v) != str) and np.isnan(v):
                    self.interactions.__setattr__(k, np.zeros((max_len, max_len)) * np.nan)

    def load(self, filepath, quiet=True) -> Tuple[np.ndarray, str, list]:
        filetype = determine_file_format(filepath)

        self.name = self.isolate_circuit_name(filepath, filetype)
        if filetype == 'csv':
            try:
                matrix, sample_names = load_csv(
                    filepath, load_as='numpy', return_header=True)
            except FileNotFoundError as ee:
                try:
                    matrix, sample_names = load_csv(
                        os.path.join(PACKAGE_DIR, filepath), load_as='numpy', return_header=True)
                except FileNotFoundError as e:
                    raise FileNotFoundError(ee)
        else:
            raise TypeError(
                f'Unsupported filetype {filetype} for loading {filepath}')
        if self.experiment_config is not None:
            units = load_units(
                filepath, experiment_config=self.experiment_config, quiet=quiet)
        else:
            units = 'unknown'
        return matrix, units, sample_names

    def isolate_circuit_name(self, circuit_filepath, filetype):
        circuit_name = None
        for faddon in INTERACTION_FIELDS_TO_WRITE:
            if faddon in circuit_filepath:
                base_name = os.path.basename(circuit_filepath).replace('.'+filetype, '').replace(
                    faddon+'_', '').replace('_'+faddon, '')
                circuit_name = base_name if type(
                    base_name) == str and faddon not in base_name else circuit_name
        if circuit_name is None:
            logging.warning(
                f'Could not find circuit name in {circuit_filepath}')
        return circuit_name

    def get_stats(self, interaction_attr='eqconstants'):
        stats = b_get_stats([self])
        return stats

    def get_interacting_species(self, idxs_interacting):
        return [idx for idx in idxs_interacting if len(set(idx)) > 1]

    def get_selfinteracting_species(self, idxs_interacting):
        return [idx for idx in idxs_interacting if len(set(idx)) == 1]

    def get_unique_interacting_idxs(self):
        idxs_interacting = np.argwhere(self.interactions.energies < 0)
        # Assuming symmetry in interactions
        idxs_interacting = sorted([tuple(sorted(i)) for i in idxs_interacting])
        return list(set(idxs_interacting))


class InteractionDataHandler():

    def __init__(self, simulation_handler: RawSimulationHandling,
                 quantities: np.ndarray):
        self.sample_processor = simulation_handler.get_sim_interpretation_protocol()
        self.sample_postproc = simulation_handler.get_postprocessing(
            initial=quantities)
        self.units = simulation_handler.units

    def init_data(self, data: dict, debug_mode: bool = False):
        if not debug_mode:
            interactions = self.parse(data)
        else:
            nans = np.ones((len(data), len(data))) * np.nan
            interactions = MolecularInteractions(
                binding_rates_association=nans,
                binding_rates_dissociation=nans,
                energies=nans,
                eqconstants=nans,
                binding_sites=nans,
                raw=data
            )
        interactions.units = self.units
        return interactions

    def parse(self, data: dict) -> MolecularInteractions:
        eqconstants, energies, a_rates, d_rates, binding = self.make_matrix(
            data)
        raw = {s.name: v for s, v in data.items()}
        return MolecularInteractions(
            binding_rates_association=a_rates,
            binding_rates_dissociation=d_rates,
            energies=energies,
            eqconstants=eqconstants,
            binding_sites=binding,
            raw=raw)

    def make_matrix(self, data: dict):
        energies = np.zeros((len(data), len(data)))
        binding = np.array([[''] * len(data)] * len(data)).astype(object)
        for i, (name_i, sample) in enumerate(data.items()):
            for j, (name_j, raw_sample) in enumerate(sample.items()):
                fields = self.sample_processor(raw_sample)
                energies[i, j] = fields['energies']
                binding[i, j] = fields['binding']
        eqconstants, (a_rates, d_rates) = self.sample_postproc(energies)
        return eqconstants, energies, a_rates, d_rates, binding


class InteractionSimulator():
    def __init__(self, sim_args: dict, allow_self_interaction: bool = True):

        self.simulation_handler = RawSimulationHandling(sim_args)
        self.simulator = self.simulation_handler.get_simulator(
            allow_self_interaction)

    def run(self, input: Union[dict, Tuple[str, dict]], quantities, compute_by_filename: bool):
        """ Makes nested dictionary for querying interactions as 
        {sample1: {sample2: interaction}} """
        self.data_handler = InteractionDataHandler(
            simulation_handler=self.simulation_handler, quantities=quantities)
        raw = self.simulator(input, compute_by_filename=compute_by_filename)
        interactions = self.data_handler.init_data(raw)
        return interactions


def b_get_stats(interactions_mxs: List[InteractionMatrix]):
    b_interaction_attrs = {}
    for interaction_attr in INTERACTION_FIELDS_TO_WRITE:
        b_interaction_attrs[interaction_attr] = np.concatenate([np.expand_dims(
            im.interactions.__getattribute__(interaction_attr), axis=0) for im in interactions_mxs], axis=0)

    batch_dim = b_interaction_attrs['energies'].ndim - 2 - 1
    idxs_interacting = get_unique_interacting_idxs(
        b_interaction_attrs['energies'], batch_dim)

    idxs_other_interacting = get_interacting_species(idxs_interacting)
    idxs_self_interacting = get_selfinteracting_species(idxs_interacting)

    idxs_other_interacting = [idxs_other_interacting[(
        idxs_other_interacting[:, :batch_dim+1] == i).flatten()][:, batch_dim+1:] for i in range(len(interactions_mxs))]
    idxs_self_interacting = [idxs_self_interacting[(idxs_self_interacting[:, :batch_dim+1] == i).flatten(
    )][:, batch_dim+1:] for i in range(len(interactions_mxs))]

    stats = {
        "name": [i.name for i in interactions_mxs],
        "interacting": idxs_other_interacting,
        "self_interacting": idxs_self_interacting,
        # "sample_names": [i.sample_names for i in interactions_mxs],
        "num_interacting": [len(i) for i in idxs_other_interacting],
        "num_self_interacting": [len(i) for i in idxs_self_interacting]
    }

    sample_names = interactions_mxs[0].sample_names
    if sample_names is None:
        sample_names = interactions_mxs
    for interaction_attr in INTERACTION_FIELDS_TO_WRITE:
        for i, s in enumerate(sample_names):
            for ii, s in enumerate(sample_names):
                attr = b_interaction_attrs[interaction_attr][:, i, ii]
                if interaction_attr == 'binding_sites' and (type(attr) == str):
                    if type(attr) == np.ndarray and len(attr) == 1:
                        string_to_tuple_list(attr[0])
                stats[interaction_attr + '_' + str(i) + '-' + str(
                    ii)] = attr
    stats = pd.DataFrame.from_dict(stats, dtype=object)
    return stats


def get_interacting_species(idxs_interacting: np.ndarray):
    assert idxs_interacting.ndim == 2, f'Array of indices should be 2-D: {idxs_interacting}'
    return idxs_interacting[idxs_interacting[:, -1] != idxs_interacting[:, -2], :]


def get_selfinteracting_species(idxs_interacting: np.ndarray):
    assert idxs_interacting.ndim == 2, f'Array of indices should be 2-D: {idxs_interacting}'
    return idxs_interacting[idxs_interacting[:, -1] == idxs_interacting[:, -2], :]


def get_unique_interacting_idxs(interaction_attr: np.ndarray, batch_dim: int):
    idxs_interacting = np.argwhere(interaction_attr < 0)
    samples = idxs_interacting[:, batch_dim+1:]
    samples.sort(axis=1)  # In-place sorting of idxs_interacting
    # idxs_interacting = np.concatenate(
    #     (idxs_interacting[:, :batch_dim], samples), axis=1)
    idxs_interacting = np.unique(idxs_interacting, axis=0)
    return idxs_interacting
    # Assuming symmetry in interactions
    # idxs_interacting = sorted([tuple(sorted(i)) for i in idxs_interacting])
    # return list(set(idxs_interacting))
