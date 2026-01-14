
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
import operator
from functools import partial
import os

import pandas as pd

from fire import Fire
from synbio_morpher.srv.io.manage.script_manager import script_preamble, visualisation_script_protocol_preamble
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.misc.scripts_io import get_search_dir
from synbio_morpher.utils.misc.string_handling import prettify_keys_for_label
from synbio_morpher.utils.results.analytics.naming import get_true_names_analytics
from synbio_morpher.utils.results.experiments import Experiment, Protocol
from synbio_morpher.utils.results.result_writer import ResultWriter
from synbio_morpher.utils.results.visualisation import visualise_data


def main(config=None, data_writer: ResultWriter = None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "synbio_morpher", "scripts", "vis_0_histplots", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    config_file, source_dirs = get_search_dir(
        config_searchdir_key='source_dirs', config_file=config_file)
    if type(source_dirs) != list:
        source_dirs = [source_dirs]

    protocols = visualisation_script_protocol_preamble(source_dirs)

    # Binding rates interactions og min and max
    def interaction_vis(data: pd.DataFrame, data_writer: ResultWriter, inputs: list):
        # Analytics visualisation
        analytics_types = get_true_names_analytics(data)

        num_mutations = list(data['mutation_num'].unique()) + ['all']
        log_opts = [(False, False), (True, False)]

        for exclude_signal in [False, True]:
            df = data
            signal_text = ''
            if exclude_signal and inputs is not None:
                signal_text = f', signal {inputs[0]} excluded'
                for o in inputs:
                    df = df[df['sample_name'] != o]
            for log_opt in log_opts:
                log_text = ''
                if any(log_opt):
                    log_text = '_log'
                for analytics_type in analytics_types:
                    for m in num_mutations:
                        if m == 'all':
                            plot_grammar_m = 's'
                            hue = 'mutation_num'
                            selection_conditions = None
                        else:
                            plot_grammar_m = 's' if m > 1 else ''
                            hue = None
                            selection_conditions = [
                                ('mutation_num', operator.eq, m)]

                        visualise_data(
                            data=df,
                            data_writer=data_writer, cols_x=[analytics_type],
                            plot_type='histplot',
                            out_name=f'{analytics_type}{log_text}_m{m}',
                            selection_conditions=selection_conditions,
                            log_axis=log_opt,
                            use_sns=True,
                            hue=hue,
                            title=f'{prettify_keys_for_label(analytics_type)}: circuit\nand mutated counterparts, {m} mutation{plot_grammar_m}' +
                            signal_text,
                            xlabel=prettify_keys_for_label(analytics_type),
                            misc_histplot_kwargs={'element': 'step'}
                        )

    protocols.append(
        Protocol(
            partial(
                interaction_vis,
                inputs=None,
                data_writer=data_writer
            ),
            req_input=True,
            req_output=True,
            name='visualise_interactions_analytics'
        )
    )

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
