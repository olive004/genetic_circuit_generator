
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from functools import partial
import os
import pandas as pd

from fire import Fire
from synbio_morpher.srv.io.manage.script_manager import script_preamble, visualisation_script_protocol_preamble
from synbio_morpher.utils.misc.scripts_io import get_search_dir
from synbio_morpher.utils.misc.string_handling import prettify_keys_for_label
from synbio_morpher.utils.results.analytics.naming import get_true_names_analytics

from synbio_morpher.utils.results.experiments import Experiment, Protocol
from synbio_morpher.utils.results.visualisation import visualise_data
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "synbio_morpher", "scripts", "vis_2_histplots_diffs_ratios_nosig", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    config_file, source_dirs = get_search_dir(
        config_searchdir_key='source_dirs', config_file=config_file)
    if type(source_dirs) != list:
        source_dirs = [source_dirs]

    protocols = visualisation_script_protocol_preamble(source_dirs)

    def vis_analytics(data: pd.DataFrame, inputs: list):
        # Analytics visualisation
        for o in inputs:
            data = data[data['sample_name'] != o]
        analytics_types = get_true_names_analytics(data)

        for analytics_type in analytics_types:
            title = prettify_keys_for_label(analytics_type)
            if 'diff' in analytics_type or 'ratio' in analytics_type:
                title = f'{prettify_keys_for_label(analytics_type)} difference between circuit\nand mutated counterparts' \
                    if 'diff' in analytics_type else f'{prettify_keys_for_label(analytics_type)} ratio from mutated\nto original circuit'
            for filltype in ['dodge', 'fill']:
                visualise_data(
                    data=data,
                    data_writer=data_writer, cols_x=[analytics_type],
                    plot_type='histplot',
                    hue='mutation_num',
                    out_name=f'{analytics_type}_log_{filltype}',
                    exclude_rows_zero_in_cols=['mutation_num'],
                    misc_histplot_kwargs={
                        "multiple": filltype,
                        "hue": 'mutation_num',
                        "element": "step"
                    },
                    log_axis=(True, False),
                    use_sns=True,
                    title=title + f', signal {inputs[0]} excluded',
                    xlabel=prettify_keys_for_label(analytics_type)
                )

    protocols.append(
        Protocol(
            partial(
                vis_analytics,
                inputs=config["signal"]["inputs"]
            ),
            req_input=True,
            name='visualise_analytics',
        )
    )

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
