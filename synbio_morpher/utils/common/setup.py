
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
import logging
from copy import deepcopy
from typing import List, Union, Optional
from synbio_morpher.srv.io.manage.data_manager import DataManager
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.misc.units import per_mol_to_per_molecule
from synbio_morpher.utils.misc.io import isolate_filename
from synbio_morpher.utils.misc.string_handling import make_circuit_name
from synbio_morpher.utils.misc.type_handling import cast_all_values_as_list
from synbio_morpher.utils.signal.configs import get_signal_type, parse_sig_args
from synbio_morpher.utils.signal.signals_new import Signal
from synbio_morpher.utils.circuit.common.config_setup import parse_cfg_args, get_configs
from synbio_morpher.utils.circuit.common.system_setup import get_system_type


ESSENTIAL_KWARGS = [
    # "interactions",
    "interaction_simulator",
    "molecular_params",
    # "mutations_args",
    "system_type",
]


def expand_model_config(in_config: dict, out_config: dict, sample_names: Union[List[str], dict]) -> dict:
    if 'starting_concentration' not in out_config.keys():
        out_config['starting_concentration'] = {}
        for s in sample_names:
            out_config['starting_concentration'][s] = in_config['molecular_params'].get(
                'starting_copynumbers', 100)
    if in_config.get('interactions') is not None or in_config.get('interactions_loaded'):
        out_config['interactions_state'] = 'loaded'
    else:
        out_config['interactions_state'] = 'uninitialised'
    return out_config


def process_molecular_params(params: dict, factor=1) -> dict:
    new_params = {}
    for k, v in params.items():
        if 'rate' in k and '_per_molecule' not in k:
            new_params[k + '_per_molecule'] = per_mol_to_per_molecule(v)
    params.update(new_params)
    if not len(params):
        raise ValueError('Need to provide molecular parameters in the config `molecular_params` field.')
    params['creation_rate'] = params['creation_rate'] * factor
    params['degradation_rate'] = params['degradation_rate'] * factor
    return params


def compose_kwargs(config: dict, prev_configs: Optional[dict] = None) -> dict:
    """ Extra configs like data paths can be supplied here, eg. for circuits that were dynamically generated. """

    if prev_configs is not None:
        for kwarg, c in prev_configs.items():
            if kwarg not in ['experiment', 'mutations_args', 'signal', 'simulation', 'include_prod_deg']:
                config[kwarg] = c

    data_manager = DataManager(filepath=config.get("data_path", None),
                               identities=config.get("identities", {}),
                               data=config.get("data", None))
    kwargs = {}
    kwargs = expand_model_config(config, kwargs, data_manager.data.sample_names)
    kwargs_specific = {
        "data": data_manager.data,
        "data_path": data_manager.source,
        # For pre-loading interactions
        "include_prod_deg": config["include_prod_deg"],
        "interactions": config["interactions"],
        "interactions_loaded": config["interactions_loaded"],
        "interaction_simulator": config["interaction_simulator"],
        "molecular_params": config["molecular_params"],
        "mutations_args": cast_all_values_as_list(config["mutations_args"]),
        "name": config.get('name', isolate_filename(data_manager.data.source)),
        "signal": config["signal"],
        "simulation": config["simulation"],
        "simulation_steady_state": config["simulation_steady_state"],
        "system_type": config["system_type"]
    }
    kwargs.update(kwargs_specific)
    kwargs["name"] = kwargs["name"] if kwargs["name"] is not None else make_circuit_name()
    assert all([e in kwargs for e in ESSENTIAL_KWARGS]), 'Some of the kwargs for composing ' \
        f'a circuit are not listed in essential kwargs {ESSENTIAL_KWARGS}: {dict({e: e in kwargs for e in ESSENTIAL_KWARGS})}'
    return kwargs


def expand_config(config: dict) -> dict:
    config["include_prod_deg"] = config.get("include_prod_deg", True)
    config["interactions"] = config.get("interactions")  # Paths to interactions
    config["interactions_loaded"] = config.get("interactions_loaded")  # Actual matrix of interactions
    config["interaction_simulator"] = config.get("interaction_simulator", {"name": "IntaRNA"})
    config["molecular_params"] = process_molecular_params(deepcopy(load_json_as_dict(config.get("molecular_params"))), config.get("molecular_params_factor", 1))
    config["signal"] = load_json_as_dict(config.get("signal"))
    return config


def add_empty_fields(config: dict) -> dict:
    config["mutations_args"] = config.get("mutations_args", {})
    config["simulation_steady_state"] = config.get("simulation_steady_state", {})
    config["simulation"] = config.get("simulation", {})
    config["system_type"] = config.get("system_type")
    return config


def prepare_config(config_filepath: Optional[Union[str, dict]] = None, config_file: Optional[dict] = None):
    config_file = get_configs(config_file, config_filepath)
    config_file = expand_config(config_file)
    config_file = parse_cfg_args(config_file)
    config_file = add_empty_fields(config_file)
    return config_file

def construct_circuit_from_cfg(prev_configs: Optional[dict], config_file: dict):
    kwargs = compose_kwargs(prev_configs=prev_configs, config=config_file)
    circuit = instantiate_system(kwargs)
    if kwargs.get("signal"):
        signal = construct_signal(kwargs, circuit)
        circuit.signal = signal
    return circuit


def construct_signal(kwargs: dict, circuit) -> Signal:
    SignalType = get_signal_type(kwargs["signal"].get("signal_type"))
    return SignalType(**parse_sig_args(kwargs, circuit))


def instantiate_system(kwargs):
    # system_cfg_args = parse_cfg_args(kwargs)
    SystemType = get_system_type(kwargs["system_type"])
    return SystemType(kwargs)
