
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


from datetime import datetime
from typing import Tuple, Union, Optional
import logging
from functools import partial
import pandas as pd


from synbio_morpher.srv.io.manage.sys_interface import SCRIPT_DIR, PACKAGE_NAME
from synbio_morpher.utils.results.writer import DataWriter
from synbio_morpher.utils.results.result_writer import ResultWriter
from synbio_morpher.utils.misc.io import get_pathnames_from_mult_dirs
from synbio_morpher.utils.results.experiments import Protocol
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict, load_csv_mult


def import_script_func(script_name):
    # importlib
    # submodule = importlib.import_module('.'+submodule_name, module.__name__)
    # script_filepath = os.path.join(
    #     SCRIPT_DIR, script_name, f'run_{script_name}.py')
    # if not os.path.isfile(script_filepath):
    #     return None
    # script_module = __import__(
    #     convert_pathname_to_module(script_filepath), fromlist=[''])
    script_module = __import__('.'.join([PACKAGE_NAME, SCRIPT_DIR, script_name, f'run_{script_name}']), fromlist=[''])
    return getattr(script_module, 'main')


def script_preamble(config, data_writer = None, alt_cfg_filepath: Optional[str] = None, use_resultwriter=True) -> Tuple[dict, Union[DataWriter, ResultWriter]]:
    Writer = ResultWriter if use_resultwriter else DataWriter
    if config is None:
        config = alt_cfg_filepath
    config_file = load_json_as_dict(config)
    if data_writer is None:
        data_writer = Writer(purpose=config_file['experiment']['purpose'])
    config_file['config_filepath_ifgiven_POSTERITY'] = alt_cfg_filepath
    return config_file, data_writer


def visualisation_script_protocol_preamble(source_dirs: list):
    return [
        Protocol(
            partial(
                get_pathnames_from_mult_dirs,
                search_dirs=source_dirs,
                file_key='tabulated_mutation_info.csv',
                first_only=True),
            req_output=True,
            name='get_pathnames_from_mult_dirs'
        ),
        Protocol(
            partial(
                load_csv_mult
                # as_type=pd.DataFrame
            ),
            req_input=True,
            req_output=True,
            name='load_csv'
        ),
        Protocol(
            partial(
                pd.concat,
                axis=0,
                ignore_index=True),
            req_input=True,
            req_output=True,
            name='concatenate_dfs'
        )
    ]


class Ensembler():

    def __init__(self, data_writer: ResultWriter, config: str) -> None:
        self.data_writer = data_writer

        self.config = load_json_as_dict(config)
        self.ensemble_configs = self.config["base_configs_ensemble"]
        self.subscripts = [script for script in self.ensemble_configs.keys()]
        self.start_time = datetime.now()

    def run(self):
        self.config['script_state'] = 'incomplete'
        self.write()
        for script_name in self.subscripts:
            script = import_script_func(script_name)
            if not script:
                continue
            logging.warning(f'\tRunning script {script_name}\n')
            config = self.ensemble_configs[script_name]
            if config["experiment"]["purpose"] != script_name:
                logging.warning(
                    f'The current script being run {script_name} does not match its '
                    f'config purpose {config["experiment"]["purpose"]}')
            self.data_writer.update_ensemble(
                config["experiment"]["purpose"])
            output = script(config, self.data_writer)
            if output:
                config, self.data_writer = output
                self.ensemble_configs[script_name] = config
            self.config["base_configs_ensemble"] = self.ensemble_configs
            self.write()

        self.data_writer.reset_ensemble()
        self.config['total_time'] = str(datetime.now() - self.start_time)
        self.config['script_state'] = 'completed'
        self.write()

    def write(self):
        self.data_writer.output(
            out_type='json', out_name='ensemble_config', data=self.config, write_master=False,
            write_to_top_dir=True, overwrite=True)



def ensemble_func(config, data_writer):

    config_file = load_json_as_dict(config)

    if data_writer is None:
        data_writer = ResultWriter(
            purpose=config_file.get('experiment', {}).get('purpose'))

    ensembler = Ensembler(data_writer=data_writer, config=config)

    ensembler.run()