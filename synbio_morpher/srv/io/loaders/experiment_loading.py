
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


import logging
from typing import Union
from typing import Optional
from synbio_morpher.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.misc.errors import ExperimentError
from synbio_morpher.utils.misc.scripts_io import get_root_experiment_folder, load_experiment_config
from synbio_morpher.utils.misc.units import per_mol_to_per_molecule


INTERACTION_FILE_ADDONS = {
    # 'coupled_binding_rates': SIMULATOR_UNITS['IntaRNA']['rate'],
    'binding_rates_dissociation': SIMULATOR_UNITS['IntaRNA']['rate'],
    'eqconstants': 'eqconstants',
    'energies': 'kcal/mol'
}


def load_param(param, experiment_config: Optional[dict] = None, filepath: Optional[str] = None) -> dict:
    if experiment_config is None:
        if filepath is not None:
            experiment_config = load_experiment_config(
                experiment_folder=get_root_experiment_folder(filepath))
        else:
            raise ValueError(f'Please supply a filepath (instead of `{filepath}`) if no experimental config is supplied.')
    return per_mol_to_per_molecule(load_json_as_dict(
        experiment_config['molecular_params'])[param])


def load_units(filepath, experiment_config: Union[dict, None] = None, quiet: bool = False) -> str:

    units = ''
    if experiment_config is None:
        try:
            experiment_config = load_experiment_config(
                experiment_folder=get_root_experiment_folder(filepath))

        except (FileNotFoundError, ExperimentError):
            if not quiet:
                logging.warning(
                    f'Units unknown - supply a valid experiment directory instead of {filepath}')
            units = 'unknown'
            return units

    simulator_cfgs = experiment_config.get('interaction_simulator', {})

    if any([i for i in INTERACTION_FILE_ADDONS.keys() if i in filepath]):
        for i, u in INTERACTION_FILE_ADDONS.items():
            if i in filepath:
                return u
    elif simulator_cfgs.get('name') == 'IntaRNA':
        if simulator_cfgs.get('postprocess'):
            units = SIMULATOR_UNITS['IntaRNA']['rate']
        else:
            units = SIMULATOR_UNITS['IntaRNA']['energy']
    else:
        units = 'unknown'
    return units
