
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from functools import partial
import os

import pandas as pd

from fire import Fire
from synbio_morpher.srv.io.manage.script_manager import script_preamble
from synbio_morpher.utils.misc.io import get_pathnames_from_mult_dirs
from synbio_morpher.utils.misc.scripts_io import get_search_dir, load_experiment_config_original
from synbio_morpher.utils.misc.string_handling import prettify_keys_for_label
from synbio_morpher.utils.misc.database_handling import expand_df_cols_lists

from synbio_morpher.utils.results.analytics.naming import get_true_names_analytics
from synbio_morpher.utils.results.experiments import Experiment, Protocol
from synbio_morpher.utils.results.visualisation import visualise_data
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict, load_csv_mult


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "synbio_morpher", "scripts", "vis_3_mutation_types_positions_1", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    config_file, source_dirs = get_search_dir(
        config_searchdir_key='source_dirs', config_file=config_file)
    if type(source_dirs) != list:
        source_dirs = [source_dirs]
    # source_dir = source_dirs[0]
    # source_config = load_experiment_config_original(
    #     source_dir, 'mutation_effect_on_interactions_signal')

    # binding_rates_threshold_upper = np.power(10, 6)
    # binding_rates_threshold_upper = None
    # binding_rates_threshold_upper_text = f', with cutoff at {binding_rates_threshold_upper}' if binding_rates_threshold_upper else ''

    protocols = [
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
            name='load_json'
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

    # Visualisations

    # Analytics visualisation
    def vis_analytics(data: pd.DataFrame):
        analytics_types = get_true_names_analytics(data)

        # Line plots
        for remove_outliers in (True, False):
            outlier_std_threshold = 3
            outlier_name = '' if remove_outliers else '_nooutliers'
            outlier_text = f', outliers removed (>{outlier_std_threshold} standard deviations)' if remove_outliers else ''

            for mutation_attr in ['mutation_positions']:
                for cols_x, cols_y, title, xlabel, ylabel in [
                        [
                            mutation_attr,
                            analytics_type,
                            f'{prettify_keys_for_label(mutation_attr)} vs. {prettify_keys_for_label(analytics_type)} difference between circuit\nand mutated counterparts',
                            f'{prettify_keys_for_label(mutation_attr)}',
                            f'{prettify_keys_for_label(analytics_type)} difference{outlier_text}'
                        ] for analytics_type in analytics_types]:

                    # Isolate the relevant columns
                    df = data[data['sample_name'] == data['sample_name'].unique()[0]]
                    df = expand_df_cols_lists(df, col_of_lists=cols_x, col_list_len='mutation_num', include_cols=[cols_y])

                    visualise_data(
                        data=df,
                        data_writer=data_writer,
                        cols_x=[cols_x], cols_y=[cols_y],
                        plot_type='line_plot',
                        out_name=f'{cols_x}_{cols_y}{outlier_name}',
                        exclude_rows_zero_in_cols=['mutation_num'],
                        log_axis=(False, True),
                        use_sns=True,
                        hue='mutation_num',
                        remove_outliers_y=remove_outliers,
                        outlier_std_threshold_y=outlier_std_threshold,
                        title=title,
                        xlabel=xlabel,
                        ylabel=ylabel
                    )

    protocols.append(
        Protocol(
            vis_analytics,
            req_input=True,
            name='visualise_interactions_difference',
            skip=config_file.get('only_visualise_circuits', False)
        )
    )

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
