
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from typing import List, Optional, Union
import logging
import pandas as pd
import numpy as np


DIFF_KEY = '_diff_to_base_circuit'
RATIO_KEY = '_ratio_from_mutation_to_base'


def get_analytics_types_base() -> List[str]:
    return ['fold_change',
            'initial_steady_states',
            'max_amount',
            'min_amount',
            'overshoot',
            'RMSE',
            'steady_states']


def get_signal_dependent_analytics() -> List[str]:
    return ['response_time',
            'precision',
            # 'precision_estimate',
            'sensitivity']  # ,
    # 'sensitivity_estimate']


def get_analytics_types_all() -> List[str]:
    return get_analytics_types() + get_analytics_types_diffs() + get_analytics_types_ratios()


def get_analytics_types() -> List[str]:
    """ The naming here has to be unique and not include small 
    raw values like diff, ratio, max, min. """
    return get_analytics_types_base() + get_signal_dependent_analytics()


def get_diffs(analytics_func=get_analytics_types) -> List[str]:
    return [a + DIFF_KEY for a in analytics_func()]


def get_ratios(analytics_func=get_analytics_types) -> List[str]:
    return [a + RATIO_KEY for a in analytics_func()]


def get_signal_dependent_analytics_all() -> List[str]:
    return get_signal_dependent_analytics() + \
        get_diffs(get_signal_dependent_analytics) + \
        get_ratios(get_signal_dependent_analytics)


def get_base_analytics_all() -> List[str]:
    return get_analytics_types_base() + \
        get_diffs(get_analytics_types_base) + \
        get_ratios(get_analytics_types_base)


def get_analytics_types_diffs() -> List[str]:
    return get_diffs(get_analytics_types)


def get_analytics_types_ratios() -> List[str]:
    return get_ratios(get_analytics_types)


def get_true_names_analytics(candidate_cols: Optional[Union[List[str], dict]]) -> List[str]:
    true_names = []
    analytics_sig = get_signal_dependent_analytics()
    analytics_base = get_base_analytics_all()

    if candidate_cols is not None:
        for c in candidate_cols:
            for s in analytics_sig:
                if s in c:
                    if (c.replace(s, '')).startswith('_wrt'):
                        # true_name = c.replace(DIFF_KEY, '').replace(RATIO_KEY, '')
                        true_names.append(c)
            for b in analytics_base:
                if b == c:
                    true_names.append(c)
    return true_names


def get_true_interaction_cols(data: pd.DataFrame, interaction_attr, remove_symmetrical=False, num_species=None) -> list:
    if num_species is None and ('sample_name' in data):
        num_species = len(data[~data['sample_name'].isna()]['sample_name'].unique())
    elif num_species is None:
        num_species = 3
        logging.warning(f'Assuming that the number of species is {num_species}')
    a = np.triu(np.ones((num_species, num_species)))
    if not remove_symmetrical:
        a += np.tril(np.ones((num_species, num_species)))
    names = list(map(lambda i: interaction_attr + '_' + str(i[0]) + '-' + str(i[1]), np.array(np.where(a > 0)).T))
        
    assert all([n in data.columns for n in names]
               ), f'Interaction info column names were not isolated correctly: {names}'
    return names
