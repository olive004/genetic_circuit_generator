
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
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
        # "synbio_morpher", "scripts", "num_interacting", "configs", "base_config_test_2.json"))
        "synbio_morpher", "scripts", "vis_2_histplots_diffs_ratios", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    config_file, source_dirs = get_search_dir(
        config_searchdir_key='source_dirs', config_file=config_file)
    if type(source_dirs) != list:
        source_dirs = [source_dirs]

    # Visualisations
    protocols = visualisation_script_protocol_preamble(source_dirs)

    def vis_analytics(data: pd.DataFrame):
        # Analytics visualisation
        analytics_types = get_true_names_analytics(data)

        # Log histplots with mutation number hue
        diff_iters = [
            [
                analytics_type,
                # f'{analytics_type}_diff_to_base_circuit',
                f'{prettify_keys_for_label(analytics_type)} difference between circuit\nand mutated counterparts',
                f'{prettify_keys_for_label(analytics_type)} difference'
            ] for analytics_type in analytics_types if 'diff' in analytics_type]
        ratio_iters = [
            [
                analytics_type,
                # f'{analytics_type}_ratio_from_mutation_to_base',
                f'{prettify_keys_for_label(analytics_type)} ratio from mutated\nto original circuit',
                f'{prettify_keys_for_label(analytics_type)} ratio'
            ] for analytics_type in analytics_types if 'ratio' in analytics_type]
        for filltype in ['dodge', 'fill']:
            for iters in diff_iters, ratio_iters:
                for cols_x, title, xlabel in iters:

                    visualise_data(
                        data=data,
                        data_writer=data_writer, cols_x=[cols_x],
                        plot_type='histplot',
                        hue='mutation_num',
                        out_name=f'{cols_x}_log_{filltype}',
                        exclude_rows_zero_in_cols=['mutation_num'],
                        misc_histplot_kwargs={
                            "multiple": filltype,
                            "hue": 'mutation_num',
                            "element": "step"
                        },
                        log_axis=(True, False),
                        use_sns=True,
                        title=title,
                        xlabel=xlabel
                    )

    protocols.append(
        Protocol(
            vis_analytics,
            req_input=True,
            name='visualise_analytics',
            skip=config_file.get('only_visualise_circuits', False)
        )
    )

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
