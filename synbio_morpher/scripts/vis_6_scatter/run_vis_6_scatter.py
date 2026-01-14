
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import os
from fire import Fire
import operator
import pandas as pd
import numpy as np

from synbio_morpher.srv.io.manage.script_manager import script_preamble, visualisation_script_protocol_preamble
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.misc.database_handling import thresh_func, select_rows_by_conditional_cols
from synbio_morpher.utils.misc.string_handling import prettify_keys_for_label
from synbio_morpher.utils.misc.scripts_io import get_search_dir
from synbio_morpher.utils.results.analytics.naming import get_true_names_analytics
from synbio_morpher.utils.results.analytics.timeseries import calculate_adaptation
from synbio_morpher.utils.results.experiments import Experiment, Protocol
from synbio_morpher.utils.results.result_writer import ResultWriter
from synbio_morpher.utils.results.visualisation import visualise_data


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "synbio_morpher", "scripts", "vis_5_means_analytics", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    # Start_experiment
    if data_writer is None:
        data_writer = ResultWriter(purpose=config_file.get(
            'experiment', {}).get('purpose', 'analyse_mutated_templates_loaded'))

    config_file, source_dirs = get_search_dir(
        config_searchdir_key='source_dirs', config_file=config_file)
    if type(source_dirs) != list:
        source_dirs = [source_dirs]

    protocols = visualisation_script_protocol_preamble(source_dirs)

    def vis(data: pd.DataFrame, data_writer, extra_naming: str = ''):
        """ Data from is multi index """

        def get_selection(m):
            if m == 'all':
                return None
            else:
                return [(
                    'mutation_num', operator.eq, m
                )]

        log_opts = [(False, False), (True, True)]

        cols = get_true_names_analytics(
            [c for c in data.columns if ('sensitivity' in c) or ('precision' in c)])
        cols = [c for c in cols if ('ratio' not in c) and ('diff' not in c)]
        cols_x = [c for c in cols if 'sensitivity' in c]
        cols_y = [c for c in cols if 'precision' in c]

        data['adaptation'] = calculate_adaptation(
            data[cols_x].to_numpy().squeeze(), data[cols_y].to_numpy().squeeze())
        # hue = 'adaptation' if (~data['adaptation'].isna()).sum() > 0 else 'overshoot'
        hues = ['adaptation', 'overshoot', 'sample_name']

        # for m in ['all']:
        for m in list(data['mutation_num'].unique()) + ['all']:
            data_selected = data
            selection_conditions = get_selection(m)
            if selection_conditions:
                data_selected = select_rows_by_conditional_cols(
                    data, selection_conditions)
                # hue = 'adaptation' if (
                #     ~data_selected['adaptation'].isna()).sum() >= int(0.5 * len(data_selected)) else 'overshoot'

            if data_selected.empty:
                continue
            
            for hue in hues:
                
                if data_selected[hue].isna().sum() == len(data_selected):
                    hue = None

                for log_opt in log_opts:

                    text_log = '_log' if log_opt[0] else ''
                    visualise_data(
                        data=data_selected,
                        data_writer=data_writer,
                        cols_x=cols_x,
                        cols_y=cols_y,
                        plot_type='scatter_plot',
                        out_name=f'adaptation_m{m}{extra_naming}{text_log}_{hue}',
                        hue=hue,
                        use_sns=True,
                        log_axis=log_opt,
                        xlabel='Sensitivity',
                        ylabel='Precision',
                        title=f'Sensitivity vs. precision',
                    )

    def visualise_all(data: pd.DataFrame, data_writer):

        vis(data, data_writer)

        # for sp in data['sample_name'].unique():
        #     vis(data[data['sample_name'] == sp], data_writer, extra_naming='_' + sp)
        #     vis(data[data['sample_name'] != sp], data_writer, extra_naming='_not-' + sp)

    protocols.append(
        Protocol(
            partial(visualise_all,
                    data_writer=data_writer,
                    ),
            req_input=True,
            name='visualise'
        )
    )

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
