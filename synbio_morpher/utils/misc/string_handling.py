
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime
from copy import deepcopy
from typing import Union, List
from difflib import SequenceMatcher
from functools import partial
import logging

import pandas as pd
import numpy as np


def add_outtype(filepath, out_type):
    if out_type in filepath:
        return filepath
    return filepath + '.' + out_type


def convert_liststr_to_list(str_list: pd.Series):
    """ Convert lists of ints that have been turned into strings
    back into lists. """
    return str_list.strip('[]').str.split(',').apply(lambda x: [int(xx) for xx in x if xx != ''])


def get_intersecting_string(string_list):
    for i in range(len(string_list)):
        match = SequenceMatcher(None, string_list[0], string_list[i]).find_longest_match(
            0, len(string_list[0]), 0, len(string_list[i]))
        base_string = string_list[0][match.a:match.b + match.size]
        if all([base_string in s for s in string_list]):
            return base_string
    raise ValueError(
        f'No intersecting string could be found in common between {string_list}')


def get_all_similar(key: str, similarity_list: list):
    similar_keys = []
    for s in similarity_list:
        if key in s:
            similar_keys.append(s)
    return similar_keys


def make_time_str():
    """Output as 'YEAR_MONTH_DAY_TIME'."""
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H%M%S")


def make_circuit_name():
    return 'circuit_' + ''.join(str(datetime.now()).replace('.', '').split(':')[1:])


def list_to_str(input_listlike):
    return ''.join(input_listlike)


def ordered_merge(list1, list2, mask) -> list:
    # its = [iter(l) for l in lists]
    # for m in mask:
    #     yield next(its[i])
    list1 = list1 if type(list1) == list else list(list1)
    merged = deepcopy(list1)
    for i, (n, c, m) in enumerate(zip(list1, list2, mask)):
        merged[i] = c if m else n
    return merged


def prettify_keys_for_label(key: str):
    key = key.replace('_', ' ')
    key = key.replace('wrt', 'with respect to')
    key = key.replace('number', 'num').replace('num', 'number')
    key = key.replace('diff', 'difference')
    key = key.replace('std', 'standard deviation')
    key = key.replace('eqconstants', 'Equilibrium constant')
    key = key.replace('binding_rates_dissociation',
                      'Dissociation rate ' + r'$k_d$' + ' (' r'$s^{-1}$' + ')')
    key = key.capitalize()
    return key


def prettify_keys_for_label_list(list_like: Union[str, list]):
    if type(list_like) == str:
        list_like = prettify_keys_for_label(list_like)
    elif type(list_like) == list:
        for i, v in enumerate(list_like):
            list_like[i] = prettify_keys_for_label_list(v)
        if len(list_like) == 1:
            list_like = list_like[0]
    return list_like


def prettify_logging_info(loggin):
    str_out = loggin
    if type(loggin) == list:
        logging.info('Printing list more legibly:')
        str_out = '\n'.join(f'{elem}' for elem in loggin)
    return str_out


def remove_file_extension(filename: str) -> str:
    return '.'.join(filename.split('.')[:-1])


def remove_special_py_functions(string_list: list) -> list:
    return [s for s in string_list if '__' not in s or '.py' not in s]


def remove_element_from_list_by_substring(string_list, exclude):
    return [x for x in string_list if exclude not in x]


def sort_by_ordinal_number(string_list: list) -> list:
    base_string = get_intersecting_string(string_list)
    int_equivalents = string_to_num_fromlist(string_list, base_string)
    sorted_strings = deepcopy(string_list)
    for i, int_equiv in enumerate(int_equivalents):
        if type(int_equiv) != int:
            raise ValueError(f'The int string was not returned as an int, but as {int_equiv} ({type(int_equiv)})')    
        sorted_strings[int_equiv-1] = string_list[i]
    return sorted_strings


def string_to_num(string_in, base_string):
    num_out = string_in.split(base_string)

    if np.char.isdigit(num_out[0]) and np.char.isdigit(num_out[1]):
        raise ValueError(
            f'Not sure which number to select in string {string_in}')
    elif np.char.isdigit(num_out[0]):
        probable_num = int(num_out[0])
    elif np.char.isdigit(num_out[1]):
        probable_num = int(num_out[1])
    else:
        probable_num = None
    return probable_num


def string_to_num_fromlist(string_list, base_string):
    return list(map(partial(string_to_num, base_string=base_string), string_list))


def string_to_tuple_list(string):
    """ GC """
    if not string or type(string) != str:
        return []
    tuple_strings = string.split(':')
    tuples = [tuple(map(int, ts.strip('()').split(',')))
              for ts in tuple_strings]
    return tuples
