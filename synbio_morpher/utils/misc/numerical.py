
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from math import factorial
from typing import Tuple, Union, List
import numpy as np
import jax.numpy as jnp
import re
import pandas as pd


NUMERICAL = {
    'infinity': np.multiply(10, 10),
    'nan': 0
}


def add_recursively(starting: list):
    if len(starting) <= 1:
        return starting
    elif len(starting) == 2:
        return np.add(starting[0], starting[1])
    else:
        return np.add(starting[0], add_recursively(starting[1:]))


def cast_astype(list_like, dtypes):
    if type(dtypes) == list:
        dtype = dtypes[0]
        if len(dtypes) == 1:
            dtypes = dtypes[0]
        else:
            dtypes = dtypes[1:]
        list_like = cast_astype(list_like, dtypes)
    else:
        dtype = dtypes
    if type(list_like) == list:
        recast = map(dtype, list_like)
    elif type(list_like) == pd.Series:
        def replace_string_list(e):
            if type(e) == str and re.search("\[.*\]", e):
                # return [dtype(ll) for ll in e[1:-1].split(',')]
                return [dtype(i) for i in e[1:-1].split(',')]
            elif type(e) == list:
                # return [dtype(ee) for ee in e]
                return dtype(e[0])
            else:
                return dtype(e)
        recast = list_like.apply(replace_string_list)
        # recast = list_like.astype(dtype)
    else:
        raise TypeError(
            f'Type {type(list_like)} cannot be converted to {dtype}')
    return recast


def binary_arpeggiator(sequence: str, count: int):
    length = len(sequence)
    interval = int(length / count)
    arpeggiation = np.arange(0, length, interval)

    for c in range(count):
        seq = np.zeros(length)
        seq[arpeggiation] = 1


def calculate_num_decimals(floatlike):
    return int(np.log10(float(str(floatlike)[::-1]))) + 1


def find_monotonic_group_idxs(lst: list, increasing=True) -> list:
    """
    Returns the number of groups of integers in lst that are monotonically increasing or decreasing without a gap.
    GC 50%

    Args:
        lst (list): A list of integers.
        increasing (bool): If True, counts groups that are monotonically increasing. If False, counts groups that are
            monotonically decreasing.

    Returns:
        int: The number of groups that match the specified condition.
    """

    if not lst:
        return []

    if increasing:
        sign = 1
    else:
        sign = -1

    all_l = []
    current_group = [lst[0]]

    if len(lst) > 1:

        for i in range(1, len(lst)):
            if sign * (lst[i] - lst[i-1]) == 1:
                current_group.append(lst[i])
            else:
                all_l.append(current_group)
                current_group = [lst[i]]
        all_l.append(current_group)

    elif len(lst) == 1:
        all_l.append(current_group)

    return all_l


def count_monotonic_group_lengths(lst, increasing=True) -> list:
    all_l = find_monotonic_group_idxs(lst, increasing)
    return [len(l) for l in all_l]


def expand_matrix_triangle_idx(flat_triangle_idx: int, flat_triangle_size: int) -> tuple:
    """ Computes the indices of a triangle, or the lower half
    of a symmetrical square matrix. Assuming that 1D index 
    traverses columns then rows.
    For example, a 1D flat input of 4 (assuming 0-indexing) would index into the 
    3rd row and 2nd column, returning the 0-based indices (2, 1) """

    symmat = make_symmetrical_matrix_from_sequence(
        np.arange(flat_triangle_size), flat_triangle_size)
    idxs = np.where(symmat == flat_triangle_idx)

    return (idxs[0], idxs[1])


def flatten_to_matrix(flattened_vector):
    length = symmetrical_matrix_length(len(flattened_vector))
    matrix = [[0] * length for _ in range(length)]

    index = 0
    for i in range(length):
        for j in range(i, length):
            matrix[i][j] = flattened_vector[index]
            matrix[j][i] = flattened_vector[index]
            index += 1

    return np.array(matrix)


def generate_mixed_binary(length: int, count: int, zeros_to_ones: bool = True):
    # TODO: This could be much better. Generate
    # sequences in a way that
    # 1. maximizes sequence orthogonality
    # 2. prioritises different neighbors
    assert count <= length, f'Currently can only produce as many sequences as length of base ' \
        'sequence. {count} sequences too many for length {length}.'
    interval = int(length / count)
    all_sequences = np.zeros((count, length)) \
        if zeros_to_ones else np.ones((count, length))
    for c in range(count):
        idxs = np.arange(c, length, count)
        all_sequences[c, idxs] = 1 if zeros_to_ones else 0

    return all_sequences


def make_dynamic_indexer(desired_axis_index_pairs: dict) -> tuple:
    """ For numpy array advanced indexing: if you know the desired index-axis pair, 
    but the axis is dynamic, use this function to create the appropriate indexing tuple """
    idxs = [0] * len(desired_axis_index_pairs)
    for axis, index in desired_axis_index_pairs.items():
        idxs[axis] = index
    return tuple(idxs)


def nan_to_replacement_string(obj, replacement_string=''):
    """ GC """
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(nan_to_replacement_string(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: nan_to_replacement_string(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return np.where(np.isnan(obj), replacement_string, obj).tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.applymap(nan_to_replacement_string)
    elif isinstance(obj, pd.Series):
        return obj.apply(nan_to_replacement_string)
    elif np.isnan(obj):
        return replacement_string
    else:
        return obj


def init_matrices(self, uniform_vals, ndims=2, init_type="rand") -> Tuple[np.ndarray]:
    matrices = (self.init_matrix(ndims, init_type, val)
                for val in uniform_vals)
    return tuple(matrices)


def init_matrix(self, ndims=2, init_type="rand", uniform_val=1) -> np.ndarray:
    matrix_size = np.random.randint(5) if self.data is None \
        else self.data.size
    if ndims > 1:
        matrix_shape = tuple([matrix_size]*ndims)
    else:
        matrix_shape = (matrix_size, 1)

    if init_type == "rand":
        return square_matrix_rand(matrix_size)
    elif init_type == "randint":
        return np.random.randint(10, 1000, matrix_shape).astype(np.float64)
    elif init_type == "uniform":
        return np.ones(matrix_shape) * uniform_val
    elif init_type == "zeros":
        return np.zeros(matrix_shape)
    raise ValueError(f"Matrix init type {init_type} not recognised.")


def invert_onehot(onehot):
    return (onehot == 0).astype(onehot.dtype)


def is_within_range(number, range_tuple):
    return np.logical_and(range_tuple[0] <= number, number <= range_tuple[-1])


def make_symmetrical(matrix):
    # """ Thank you Copilot. Matrix should be square. """
    # return (matrix + matrix.T) / 2
    return matrix + matrix.T - np.diag(matrix.diagonal())


def make_symmetrical_matrix_from_sequence(arr: np.ndarray, side_length: int):
    """ For a flat 1D array, make a symmetrical 2D matrix filling
    in the upper triangle with the 1D array. """
    if arr.ndim > 1:
        logging.warning('For flat 1D sequence array, remember to squeeze')
    # i = np.triu(
    #     np.arange(side_length * side_length).reshape(side_length, side_length))
    # # idxs = np.interp(ii, np.unique(ii), np.arange(len(arr))).astype(int)
    # ii = i + i.T - np.diag(i.diagonal())
    
    i = np.triu(np.cumsum(np.triu(np.ones(side_length))).reshape(side_length, side_length) - 1).astype(int)
    ii = i + i.T - np.diag(i.diagonal())
    return arr[ii].reshape(side_length, side_length)


def make_symmetrical_matrix_from_sequence_nojax(arr, side_length: int):
    """ For a flat 1D array, make a symmetrical 2D matrix filling
    in the upper triangle with the 1D array. Not jax-friendly. """
    n = np.zeros((side_length, side_length))
    n[np.triu_indices(side_length)] = arr
    return n


def np_delete_axes(array, rowcol: Union[list, int], axes: list):
    for axis in axes:
        array = np.delete(array, rowcol, axis=axis)
    return array


def nPr(n, r):
    return int(factorial(n)/factorial(n-r))


def round_to_nearest(x, base):
    return base * round(x/base)


def square_matrix_rand(num_nodes: int = 3):
    dims = (num_nodes, num_nodes)
    return np.random.rand(*dims)


def symmetrical_matrix_length(flattened_length):
    return int((-1 + np.sqrt(1 + 8 * flattened_length)) / 2)


def transpose_arraylike(arraylike):
    if type(arraylike) == list:
        arraylike = np.array(arraylike)
    return np.transpose(arraylike)


def triangular_sequence(n: int) -> int:
    return int((n*(n+1))/2)


def zero_out_negs(npmatrix):
    return npmatrix.clip(min=0)


pass
