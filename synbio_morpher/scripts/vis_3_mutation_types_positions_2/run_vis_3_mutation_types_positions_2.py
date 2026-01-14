
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from functools import partial
from typing import List
import logging
import os

import pandas as pd

from fire import Fire
from synbio_morpher.srv.io.manage.script_manager import script_preamble
from synbio_morpher.utils.misc.io import get_pathnames_from_mult_dirs
from synbio_morpher.utils.misc.scripts_io import get_search_dir
from synbio_morpher.utils.misc.string_handling import prettify_keys_for_label
from synbio_morpher.utils.misc.database_handling import expand_df_cols_lists
from synbio_morpher.utils.results.analytics.naming import get_true_names_analytics

from synbio_morpher.utils.results.experiments import Experiment, Protocol
from synbio_morpher.utils.results.visualisation import visualise_data
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict, load_csv_mult


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "synbio_morpher", "scripts", "vis_3_mutation_types_positions_2", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    config_file, source_dirs = get_search_dir(
        config_searchdir_key='source_dirs', config_file=config_file)
    if type(source_dirs) != list:
        source_dirs = [source_dirs]

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

    def plot_mutation_attr(data: pd.DataFrame, mutation_attr: str, remove_outliers: bool, log_axis: tuple, plot_type: str, analytics_types: List[str]):
        outlier_std_threshold = 3
        outlier_text = f', outliers removed (>{outlier_std_threshold} standard deviations)' if remove_outliers else ''

        log_text = '_log' if any(log_axis) else ''
        outlier_save_text = '_nooutliers' if remove_outliers else ''
        df = data[data['sample_name'] == data['sample_name'].unique()[0]]
        df = expand_df_cols_lists(
            df, col_of_lists=mutation_attr, col_list_len='mutation_num', include_cols=analytics_types)
        if len(df[mutation_attr].unique()) > 1e4:
            logging.warning(f'Plot type {plot_type} may not be able to handle this many data points')
        # extra_kwargs = {
        #     'errorbar': None,
        #     'n_boot':0} if len(df[mutation_attr]) > 1e5 else {}

        for cols_y in analytics_types:

            df2 = df[['mutation_num', mutation_attr, cols_y]]
            visualise_data(
                data=df2,
                data_writer=data_writer,
                cols_x=[mutation_attr], cols_y=[cols_y],
                plot_type=plot_type,
                out_name=f'{cols_y}_{mutation_attr}{log_text}{outlier_save_text}',
                exclude_rows_zero_in_cols=['mutation_num'],
                log_axis=log_axis,
                use_sns=True,
                hue='mutation_num',
                remove_outliers_y=remove_outliers,
                outlier_std_threshold_y=outlier_std_threshold,
                title=f'{prettify_keys_for_label(mutation_attr)} vs. {prettify_keys_for_label(cols_y)}',
                xlabel=f'{prettify_keys_for_label(mutation_attr)}',
                ylabel=f'{prettify_keys_for_label(cols_y)}{outlier_text}'
            )

    def vis_data(data: pd.DataFrame):
        analytics_types = get_true_names_analytics(data.columns)
        for remove_outliers in [False, True]:
            for log_axis in [(False, False), (False, True)]:
                plot_mutation_attr(data, 'mutation_type', remove_outliers=remove_outliers, log_axis=log_axis,
                                   plot_type='bar_plot', analytics_types=analytics_types)
                plot_mutation_attr(data, 'mutation_positions', remove_outliers=remove_outliers, log_axis=log_axis,
                                   plot_type='line_plot', analytics_types=analytics_types)

    protocols.append(
        Protocol(
            vis_data,
            req_input=True,
            name='visualise'
        )
    )

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
