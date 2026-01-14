
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import Dict, List, Union, Optional

from synbio_morpher.srv.parameter_prediction.simulator_loading import find_simulator_loader
from synbio_morpher.srv.io.manage.sys_interface import PACKAGE_DIR, PACKAGE_NAME
from synbio_morpher.utils.misc.io import get_pathnames
from synbio_morpher.utils.misc.string_handling import remove_special_py_functions
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.srv.io.manage.sys_interface import make_filename_safely
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict


def get_configs(config_file: Optional[dict], config_filepath: Optional[Union[str, dict]]) -> dict:
    config_filepath = make_filename_safely(config_filepath)
    if config_file is None:
        if config_filepath is None:
            raise ValueError('Config file or path needed as input to function.')
        else:
            config_file = load_json_as_dict(config_filepath)
    else:
        if config_filepath:
            raise ValueError(
                'Both a config and a config filepath were defined - only use one config option.')
    return config_file


def create_argparse_from_dict(dict_args: dict):
    parser = argparse.ArgumentParser()
    args_namespace, left_argv = parser.parse_known_args(args=dict_args) # type: ignore
    args_namespace = update_namespace_with_dict(args_namespace, dict_args)
    return args_namespace, left_argv


def get_simulator_names() -> List:
    simulator_dir = os.path.join(PACKAGE_DIR, "srv", "parameter_prediction")
    simulators = remove_special_py_functions(os.listdir(simulator_dir))
    from synbio_morpher.srv.parameter_prediction.simulator_loading import extra_simulators
    simulators = simulators + extra_simulators
    return simulators


def handle_simulator_cfgs(simulator, simulator_cfg_path):
    simulator_cfg = load_json_as_dict(simulator_cfg_path)
    cfg_protocol = find_simulator_loader(simulator)
    if any([os.sep in v for v in simulator_cfg.values() if type(v) == str]):
        for k, v in simulator_cfg.items():
            # Make into file
            if type(v) == str and (os.sep in v):
                if os.path.isfile(v):
                    simulator_cfg[k] = v
                elif not os.path.isfile(v) and (PACKAGE_NAME in v):
                    simulator_cfg[k] = os.path.join(
                        PACKAGE_DIR, v.split(PACKAGE_NAME)[-1].strip(os.sep))
                    # simulator_cfg[k] = os.path.join(PACKAGE_DIR, v) if (PACKAGE_NAME in v) and (os.sep in v) and (PACKAGE_DIR not in v) else v
                if not os.path.isfile(simulator_cfg[k]):
                    raise ValueError(
                        f'Could not find simulator file {simulator_cfg[k]}. Check that the package directory is correct.')
    return cfg_protocol(simulator_cfg)


def parse_cfg_args(config: dict, default_args: Union[dict, None] = None) -> dict:

    default_args = retrieve_default_args() if default_args is None else default_args

    simulator_kwargs = load_simulator_kwargs(default_args, config)
    config['interaction_simulator']['simulator_kwargs'] = simulator_kwargs
    config['interaction_simulator']['molecular_params'] = config['molecular_params']
    config['interaction_simulator']['compute_by_filename'] = config['interaction_simulator'].get(
        'compute_by_filename', False)
    default_args['simulation_steady_state'].update(config.get('simulation_steady_state', {}))
    config['simulation_steady_state'] = default_args['simulation_steady_state']

    return config


def load_simulator_kwargs(default_args: dict, config_args: dict) -> Union[dict, None]:
    target_simulator_name = config_args.get(
        'interaction_simulator', {}).get('name')
    simulator_kwargs = None
    for simulator_name in get_simulator_names():
        kwarg_flag = simulator_name in default_args
        if not target_simulator_name is None:
            kwarg_flag = kwarg_flag and simulator_name == target_simulator_name
        if kwarg_flag:
            simulator_kwargs = handle_simulator_cfgs(
                simulator_name, default_args[simulator_name])
            if config_args.get('interaction_simulator', {}).get('simulator_kwargs'):
                simulator_kwargs.update(config_args['interaction_simulator']['simulator_kwargs'])
            break
    return simulator_kwargs


def retrieve_default_args() -> dict:
    fn = get_pathnames(file_key='default_args', search_dir=os.path.join(
        PACKAGE_DIR, 'utils', 'common', 'configs', 'simulators'), first_only=True)
    if (type(fn) == str) or (type(fn) == dict):
        default_args = load_json_as_dict(fn)
        return default_args
    else: raise ValueError(f'Filename(s) {fn} could not be found')


def update_namespace_with_dict(namespace_args, updater_dict: Dict):
    vars(namespace_args).update(updater_dict)
    return namespace_args
