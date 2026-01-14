
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from functools import partial
import os
import operator

import pandas as pd

from fire import Fire
from synbio_morpher.srv.io.manage.script_manager import script_preamble, visualisation_script_protocol_preamble
from synbio_morpher.utils.misc.scripts_io import get_search_dir
from synbio_morpher.utils.misc.string_handling import prettify_keys_for_label
from synbio_morpher.utils.results.analytics.naming import get_true_interaction_cols

from synbio_morpher.utils.results.experiments import Experiment, Protocol
from synbio_morpher.utils.results.visualisation import visualise_data
from synbio_morpher.srv.parameter_prediction.interactions import INTERACTION_TYPES
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "synbio_morpher", "scripts", "vis_1_histplots_interactions", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    config_file, source_dirs = get_search_dir(
        config_searchdir_key='source_dirs', config_file=config_file)
    if type(source_dirs) != list:
        source_dirs = [source_dirs]

    # Visualisations
    protocols = visualisation_script_protocol_preamble(source_dirs)

    def visualise_interactions_raw(data: pd.DataFrame, data_writer):
        # data = data[data['sample_name'] == data['sample_name'].iloc()[0]]
        log_opts = [(False, False), (True, False)]
        num_mutations = list(data['mutation_num'].unique())
        for interaction_type in INTERACTION_TYPES:
            interaction_cols = get_true_interaction_cols(
                data, interaction_type, remove_symmetrical=True)
            if not interaction_cols:
                continue
            units_text = '(' + r'$s^{-1}$' + \
                ') ' if 'rate' in interaction_type else ''

            for log_opt in log_opts:
                log_text = '_log' if any(log_opt) else ''

                for m in num_mutations + ['all'] + ['all-pooled']:
                    df = pd.concat(objs=[
                        pd.DataFrame.from_dict(
                            {interaction_type: data[interaction_col],
                             'mutation_num': data['mutation_num']}
                        ) for interaction_col in interaction_cols
                    ])
                    if 'all' in str(m):
                        plot_grammar_m = 's'
                        hue = 'mutation_num'
                        selection_conditions = None
                        if m == 'all-pooled':
                            hue = None
                    else:
                        plot_grammar_m = 's' if m > 1 else ''
                        hue = None
                        selection_conditions = [
                            ('mutation_num', operator.eq, m)]

                    visualise_data(
                        data=df,
                        cols_x=[interaction_type],
                        plot_type='histplot',
                        data_writer=data_writer,
                        out_name=interaction_type + log_text + '_m' + str(m),
                        hue=hue,
                        selection_conditions=selection_conditions,
                        log_axis=log_opt,
                        use_sns=True,
                        title=f'{prettify_keys_for_label(interaction_type)} {units_text}for {m} mutation{plot_grammar_m}',
                        misc_histplot_kwargs={'element': 'step'}
                    )

    protocols.append(
        Protocol(
            partial(
                visualise_interactions_raw,
                data_writer=data_writer
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
