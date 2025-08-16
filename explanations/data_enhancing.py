from functools import partial
import numpy as np
import os
import sys
import pandas as pd
import jax
from typing import Tuple


if __package__ is None or (__package__ == ''):

    module_path = os.path.abspath(os.path.join('..'))
    sys.path.append(module_path)
    sys.path.append(os.path.abspath(os.path.join('.')))

    __package__ = os.path.basename(module_path)


from synbio_morpher.utils.data.data_format_tools.manipulate_fasta import load_seq_from_FASTA
from synbio_morpher.utils.misc.numerical import count_monotonic_group_lengths, find_monotonic_group_idxs, is_within_range
from synbio_morpher.utils.misc.string_handling import string_to_tuple_list, convert_liststr_to_list
from synbio_morpher.utils.misc.type_handling import get_nth_elements
from synbio_morpher.utils.results.analytics.naming import get_true_interaction_cols

SEQ_LENGTH = 20


def proc_info(info):
    info['num_interacting_all'] = info['num_interacting'] + \
        info['num_self_interacting']
    info['sp_distance'] = 0
    info.loc[(info['sensitivity'] <= 1) & (info['precision'] <= 10), 'sp_distance'] = np.sqrt(
        np.power(1-info['sensitivity'], 2) + np.power(10 - info['precision'], 2))
    info.loc[(info['sensitivity'] <= 1) & (info['precision']
                                                         > 10), 'sp_distance'] = np.absolute(info['sensitivity'] - 1)
    info.loc[(info['sensitivity'] > 1) & (info['precision']
                                                        <= 10), 'sp_distance'] = np.absolute(info['precision'] - 10)

    if type(info['mutation_type'].iloc[0]) == str:
        info['mutation_type'] = convert_liststr_to_list(
            info['mutation_type'].str)
    if type(info['mutation_positions'].iloc[0]) == str:
        info['mutation_positions'] = convert_liststr_to_list(
            info['mutation_positions'].str)

    #  Binding sites

    num_group_cols = [e.replace('energies', 'binding_sites_groups')
                      for e in get_true_interaction_cols(info, 'energies')]
    num_bs_cols = [e.replace('energies', 'binding_sites_count')
                   for e in get_true_interaction_cols(info, 'energies')]
    bs_idxs_cols = [e.replace('energies', 'binding_sites_idxs')
                    for e in get_true_interaction_cols(info, 'energies')]
    bs_range_cols = [e.replace('energies', 'binding_site_group_range')
                     for e in get_true_interaction_cols(info, 'energies')]

    for b, g, bs, bsi, r in zip(get_true_interaction_cols(info, 'binding_sites'), num_group_cols, num_bs_cols, bs_idxs_cols, bs_range_cols):
        # fbs = [string_to_tuple_list(bb) for bb in info[b]]
        fbs = jax.tree_util.tree_map(
            lambda bb: string_to_tuple_list(bb), info[b].to_list())
        first = get_nth_elements(fbs, empty_replacement=[])
        info[bs] = list(map(count_monotonic_group_lengths, first))
        info[bsi] = list(map(find_monotonic_group_idxs, first))
        info[g] = info[bs].apply(len)
        # info[r] = [[(bb[0], bb[-1]) for bb in b] for b in info[bsi]]
        info[r] = list(
            map(lambda b: list(map(lambda x: (x[0], x[-1]), b)), info[bsi]))

    # Mutation number ratiometric change

    numerical_cols = [c for c in info.columns if (type(info[(info['mutation_num'] > 0) & (info['eqconstants_0-0'] > 1)][c].iloc[0]) != str) and (
        type(info[c].iloc[0]) != list) and c not in get_true_interaction_cols(info, 'binding_sites')]
    key_cols = ['circuit_name', 'interacting',
                'mutation_name', 'name', 'sample_name']

    grouped = info.groupby(['circuit_name', 'sample_name'], as_index=False)
    mutation_log = grouped[numerical_cols].apply(
        lambda x: np.log(x / x.loc[x['mutation_num'] == 0].squeeze()))
    mutation_log = mutation_log.reset_index()
    mutation_log[numerical_cols] = info[numerical_cols]
    info[[c + '_logm' for c in numerical_cols]] = mutation_log[numerical_cols]

    return info, num_group_cols, num_bs_cols, numerical_cols, key_cols, mutation_log, bs_range_cols


# Melt energies

def melt(info: pd.DataFrame, num_group_cols: list, num_bs_cols: list, numerical_cols: list, key_cols: list, mutation_log: pd.DataFrame, bs_range_cols: list, num_species=3, include_log=False):
    get_true_interaction_cols2 = partial(
        get_true_interaction_cols, num_species=num_species)
    good_cols = list(info.columns)
    [good_cols.remove(x) for x in get_true_interaction_cols2(info, 'binding_rates_dissociation') + get_true_interaction_cols2(info, 'eqconstants') +
     get_true_interaction_cols2(info, 'energies') + get_true_interaction_cols2(info, 'binding_sites') + num_group_cols + num_bs_cols]
    binding_idx_map = {e.replace('energies_', ''): i for i, e in enumerate(
        get_true_interaction_cols2(info, 'energies'))}

    infom = info.melt(good_cols, value_vars=get_true_interaction_cols2(
        info, 'energies'), var_name='idx', value_name='energies')
    dfm = info.melt(good_cols, value_vars=num_group_cols,
                    var_name='num_groups_idx', value_name='num_groups')
    infom['idx_species_binding'] = dfm['num_groups_idx'].apply(
        lambda x: binding_idx_map[x.replace('binding_sites_groups_', '')])
    infom['num_groups'] = dfm['num_groups']
    dfm = info.melt(good_cols, value_vars=num_bs_cols,
                    var_name='num_bs_idx', value_name='num_bs')
    infom['num_bs'] = dfm['num_bs']

    mutation_cols = [c for c in numerical_cols + key_cols if (c not in get_true_interaction_cols2(mutation_log, 'energies') +
                                                              get_true_interaction_cols2(mutation_log, 'binding_rates_dissociation') +
                                                              get_true_interaction_cols2(mutation_log, 'eqconstants') +
                                                              get_true_interaction_cols2(mutation_log, 'binding_sites_groups')) and (
        c in mutation_log.columns
    )]
    if include_log:
        infom['energies' + '_logm'] = mutation_log.melt(mutation_cols, value_vars=get_true_interaction_cols2(
            mutation_log, 'energies'), var_name='energies_idx', value_name='energies')['energies']

    for k in ['binding_sites', 'binding_rates_dissociation', 'eqconstants']:
        infom[k] = info.melt(good_cols, value_vars=get_true_interaction_cols2(
            info, k), var_name=f'{k}_idx', value_name=k)[k]
        if include_log and (k != 'binding_sites'):
            infom[k + '_logm'] = mutation_log.melt(mutation_cols, value_vars=get_true_interaction_cols2(
                mutation_log, k), var_name=f'{k}_idx', value_name=k)[k]

    # Energy diffs:

    for k in ['binding_rates_dissociation', 'eqconstants', 'energies']:
        infom[f'{k}_diffs'] = info.groupby(['circuit_name'])[get_true_interaction_cols2(info, k)].apply(
            lambda x: x - x.iloc[0]).melt(value_vars=get_true_interaction_cols2(info, k), var_name='idx', value_name=f'{k}_diffs')[f'{k}_diffs']
        if include_log:
            infom[f'{k}_diffs' + '_logm'] = mutation_log[get_true_interaction_cols2(mutation_log, k)].apply(
                lambda x: x - x.iloc[0]).melt(value_vars=get_true_interaction_cols2(mutation_log, k), var_name='idx', value_name=f'{k}_diffs')[f'{k}_diffs']

    info_e = info.explode(column=['mutation_type', 'mutation_positions'])

    mut_in_bs_cols = [e.replace('energies', 'is_mutation_in_binding_site')
                      for e in get_true_interaction_cols2(info, 'energies')]
    mut_in_edge_cols = [e.replace('energies', 'is_mutation_on_edge')
                        for e in get_true_interaction_cols2(info, 'energies')]

    for isb, r in zip(mut_in_bs_cols, bs_range_cols):
        info_e[isb] = [any([is_within_range(m, r) for r in range_tuples])
                       for m, range_tuples in zip(info_e['mutation_positions'], info_e[r])]
    for ise, r in zip(mut_in_edge_cols, bs_range_cols):
        info_e[ise] = [any([(m == r[0]) or (m == r[-1]) for r in range_tuples])
                       for m, range_tuples in zip(info_e['mutation_positions'], info_e[r])]

    infom['num_muts_in_binding_site'] = info_e.reset_index().groupby(['index', 'circuit_name', 'mutation_name'], as_index=False).agg({isb: 'sum' for isb in mut_in_bs_cols}).melt(
        id_vars=['circuit_name', 'mutation_name'], value_vars=mut_in_bs_cols, var_name='idx', value_name='num_muts_in_binding_site')['num_muts_in_binding_site']
    infom['num_muts_in_binding_site_edge'] = info_e.reset_index().groupby(['index', 'circuit_name', 'mutation_name'], as_index=False).agg({ise: 'sum' for ise in mut_in_edge_cols}).melt(
        id_vars=['circuit_name', 'mutation_name'], value_vars=mut_in_edge_cols, var_name='idx', value_name='num_muts_in_binding_site_edge')['num_muts_in_binding_site_edge']
    # infom['frac_muts_in_binding_site'] = info_e.groupby(['circuit_name', 'mutation_num', 'mutation_name', 'sample_name'], as_index=False).agg({isb: lambda x: sum(x) / np.max([1, len(x)]) for isb in mut_in_bs_cols}).melt(
    #     id_vars=['circuit_name', 'mutation_name', 'sample_name'], value_vars=mut_in_bs_cols, var_name='idx', value_name='frac_muts_in_binding_site')['frac_muts_in_binding_site']
    infom['frac_muts_in_binding_site'] = infom['num_muts_in_binding_site'] / \
        (infom['mutation_num'].to_numpy() +
         (infom['mutation_num'] == 0).to_numpy())
    infom['frac_muts_in_binding_site_edge'] = infom['num_muts_in_binding_site_edge'] / \
        (infom['mutation_num'].to_numpy() +
         (infom['mutation_num'] == 0).to_numpy())

    return infom


def get_named_aggs(info: pd.DataFrame, include_log: bool = False) -> dict:
    # Standard Deviations

    relevant_cols = [
        'fold_change',
        # 'initial_steady_states',
        # 'max_amount', 'min_amount',
        'overshoot',
        'RMSE',
        'steady_states',
        # 'response_time',
        # 'response_time_diff_to_base_circuit',
        # 'response_time_ratio_from_mutation_to_base',
        'precision',
        'precision_diff_to_base_circuit',
        'precision_ratio_from_mutation_to_base',
        'sensitivity',
        'sensitivity_diff_to_base_circuit',
        'sensitivity_ratio_from_mutation_to_base',
        'fold_change_diff_to_base_circuit',
        # 'initial_steady_states_diff_to_base_circuit',
        # 'max_amount_diff_to_base_circuit', 'min_amount_diff_to_base_circuit',
        'overshoot_diff_to_base_circuit',
        # 'RMSE_diff_to_base_circuit',
        'steady_states_diff_to_base_circuit',
        'fold_change_ratio_from_mutation_to_base',
        # 'initial_steady_states_ratio_from_mutation_to_base',
        # 'max_amount_ratio_from_mutation_to_base',
        # 'min_amount_ratio_from_mutation_to_base',
        # 'overshoot_ratio_from_mutation_to_base',
        # 'RMSE_ratio_from_mutation_to_base',
        'steady_states_ratio_from_mutation_to_base',
        # 'num_groups',
    ]
    energy_cols = [
        'energies',
        'binding_rates_dissociation',
        'eqconstants',
        'energies_diffs',
        'binding_rates_dissociation_diffs',
        'eqconstants_diffs'
    ]
    [relevant_cols.append(c) for c in energy_cols if c in info.columns]

    named_aggs = {}
    for c in relevant_cols:
        col_iter = [c, c + '_logm'] if include_log else [c]
        for cc in col_iter:
            named_aggs.update(
                {cc + '_std': pd.NamedAgg(column=cc, aggfunc="std")})
            named_aggs.update(
                {cc + '_mean': pd.NamedAgg(column=cc, aggfunc="mean")})
            named_aggs.update({cc + '_std_normed_by_mean': pd.NamedAgg(column=cc,
                                                                       aggfunc=lambda x: np.std(x) / np.max([1, np.mean(x)]))})
    return named_aggs


def summ(info: pd.DataFrame) -> pd.DataFrame:
    named_aggs = get_named_aggs(info, include_log=False)
    info_summ = info.groupby(
        ['circuit_name', 'mutation_num', 'sample_name'], as_index=False).agg(**named_aggs)
    return info_summ


def summ_energies(infom: pd.DataFrame, include_log: bool = True) -> pd.DataFrame:
    # Mutations within binding
    named_aggs = get_named_aggs(infom, include_log=include_log)
    info_summ = infom.groupby(
        ['circuit_name', 'mutation_num', 'sample_name'], as_index=False).agg(**named_aggs)

    v = infom.groupby(['circuit_name', 'mutation_num', 'sample_name'], as_index=False).agg(
        **{'frac_muts_in_binding_site' + '_std': pd.NamedAgg(column='frac_muts_in_binding_site', aggfunc='std'),
           'frac_muts_in_binding_site' + '_mean': pd.NamedAgg(column='frac_muts_in_binding_site', aggfunc='mean'),
           'frac_muts_in_binding_site' + '_std_normed_by_mean': pd.NamedAgg(
               column='frac_muts_in_binding_site', aggfunc=lambda x: np.std(x) / np.max([1, np.mean(x)]))}) # type: ignore
    info_summ['frac_muts_in_binding_site' + '_std'
              ] = v['frac_muts_in_binding_site' + '_std']
    info_summ['frac_muts_in_binding_site' + '_mean'
              ] = v['frac_muts_in_binding_site' + '_mean']
    info_summ['frac_muts_in_binding_site' + '_std_normed_by_mean'
              ] = v['frac_muts_in_binding_site' + '_std_normed_by_mean']

    return info_summ


def enhance_data(info: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Add / process columns

    info, num_group_cols, num_bs_cols, numerical_cols, key_cols, mutation_log, bs_range_cols = proc_info(
        info)
    infom = melt(info, num_group_cols, num_bs_cols, numerical_cols,
                 key_cols, mutation_log, bs_range_cols)
    info_summ = summ(infom)

    return info, infom, info_summ


def load_circuits(data: pd.DataFrame, root_dir: str = '.'):
    if 'mutation_species' not in data.columns:
        data['mutation_species'] = data['mutation_names'].str[:4]

    mutation_species = data['mutation_species'].unique()
    circuit_name = data['circuit_name'].unique()
    path_to_template_circuit = list(
        data[data['mutation_num'] > 0]['path_to_template_circuit'].unique())
    circuit_paths = jax.tree_util.tree_map(lambda x: os.path.join(
        root_dir, 'data', 'raw', str(x.split('data/')[-1])), path_to_template_circuit)
    fastas = jax.tree_util.tree_map(
        lambda cp: load_seq_from_FASTA(cp, as_type='dict'), circuit_paths)
    fasta_d = dict(zip(circuit_name, fastas))
