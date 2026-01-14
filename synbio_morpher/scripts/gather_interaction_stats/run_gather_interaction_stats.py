
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import os

from fire import Fire

from synbio_morpher.srv.io.manage.script_manager import script_preamble
from synbio_morpher.srv.parameter_prediction.interactions import INTERACTION_FIELDS_TO_WRITE
from synbio_morpher.srv.sequence_exploration.sequence_analysis import generate_interaction_stats
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.misc.io import get_pathnames
from synbio_morpher.utils.misc.scripts_io import get_search_dir
from synbio_morpher.utils.misc.type_handling import flatten_listlike
from synbio_morpher.utils.results.experiments import Experiment, Protocol
from synbio_morpher.utils.results.writer import DataWriter


def readout(var_obj):
    logging.info(
        f'Using directory or files {var_obj} for gathering the gene circuit interactions from.')
    return var_obj


def main(config=None, data_writer=None):
    # set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "synbio_morpher", "scripts", "gather_interaction_stats", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    # start_experiment
    if data_writer is None:
        data_writer_kwargs = {'purpose': config_file.get(
            "experiment").get("purpose")}
        data_writer = DataWriter(**data_writer_kwargs)

    config_file, search_dirs = get_search_dir(
        config_searchdir_key="source_of_interactions", config_file=config_file)

    protocols = []
    if not any([os.path.isdir(os.path.join(search_dirs, i)) for i in INTERACTION_FIELDS_TO_WRITE]):
        search_dirs = [f.path for f in os.scandir(search_dirs) if f.is_dir()] + \
            flatten_listlike([[ff.path for ff in os.scandir(os.path.join(
                f.path, 'mutations'))] for f in os.scandir(search_dirs) if f.is_dir() and os.path.isdir(os.path.join(f.path, 'mutations'))])
    else:
        search_dirs = [search_dirs]

    for s in search_dirs:
        protocols.append(
            Protocol(
                # get list of all interaction paths
                partial(get_pathnames,
                        file_key=INTERACTION_FIELDS_TO_WRITE,
                        search_dir=s,
                        # subdirs=subdirs,
                        subdirs=INTERACTION_FIELDS_TO_WRITE,
                        as_dict=True,
                        ),
                req_output=True,
                name='get_pathnames'
            ))
        protocols.append(
            [
                Protocol(
                    partial(generate_interaction_stats,
                            experiment_dir=search_dirs, writer=data_writer),
                    req_output=True,
                    req_input=True,
                    name='analyse_interactions'
                )
            ]
        )
    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()

    return config, data_writer


if __name__ == "__main__":
    Fire(main)
