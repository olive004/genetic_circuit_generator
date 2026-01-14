
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Optional
# from synbio_morpher.utils.misc.type_handling import merge_dicts
from synbio_morpher.utils.results.analytics.naming import DIFF_KEY, RATIO_KEY


TIMEAXIS = 1


def calculate_adaptation(s, p, alpha=3, s_center=7, p_center=7.5):
    """ Adaptation = robustness to noise. Greater than 0 is good.
    s = sensitivity, p = precision 
    High when s > 1 and p > 10
    """
    s_log = jnp.log10(s)
    p_log = jnp.log10(p)
    a = -(alpha*(s_log - s_center)**2 + (p_log - p_center)**2) + 1000
    return jnp.where((a > -jnp.inf) & (a < jnp.inf),
                     a,
                     jnp.nan)
    
    
def compute_adaptability_full(ys_steady, ys_signal, idx_sig, use_sensitivity_func1):
    """ ts: time series with dimensions [t, species] """

    if use_sensitivity_func1:
        peaks = compute_peaks(ys_steady[-1], ys_signal[-1],
                              ys_signal.max(axis=0), ys_signal.min(axis=0))

        s = compute_sensitivity(
            signal_idx=idx_sig,
            starting_states=ys_steady[-1],
            peaks=peaks,
        )
    else:
        s = compute_sensitivity2(
            starting_states=ys_steady[-1],
            minv=ys_signal.min(axis=0),
            maxv=ys_signal.max(axis=0),
            signal_0=ys_steady[-1, idx_sig],
            signal_1=ys_signal[0, idx_sig]
        )
    p = compute_precision(
        starting_states=ys_steady[-1],
        steady_states=ys_signal[-1],
        signal_0=ys_steady[-1, idx_sig],
        signal_1=ys_signal[0, idx_sig])
    a = calculate_adaptation(s, p)
    
    return a, s, p


def compute_derivative(data):
    if data.shape[TIMEAXIS] <= 1:
        return np.ones_like(data) * np.inf
    deriv = jnp.gradient(data)[1]
    return deriv


def compute_fold_change(starting_states, steady_states):
    denom = jnp.where(starting_states != 0,
                      starting_states, -1)
    fold_change = jnp.where(denom != -1,
                            steady_states / denom, np.nan)
    return fold_change


def compute_overshoot(steady_states, peaks):
    return jnp.absolute(peaks - steady_states)


def compute_peaks_deprecated(initial_steady_states: jnp.ndarray, final_steady_states: jnp.ndarray, maxa: jnp.ndarray, mina: jnp.ndarray):
    return jnp.where(
        initial_steady_states < final_steady_states,
        maxa - mina + initial_steady_states,
        mina - maxa + initial_steady_states
    )


def compute_peaks(initial_steady_states: jnp.ndarray, final_steady_states: jnp.ndarray, maxa: jnp.ndarray, mina: jnp.ndarray):
    peak = jnp.where(final_steady_states > initial_steady_states, maxa, mina)
    return jnp.where(initial_steady_states < peak, peak, maxa if peak is mina else mina)


def calculate_precision_core(output_diff, starting_states, signal_diff, signal_0) -> jnp.ndarray:
    numer = jnp.where(signal_0 != 0, signal_diff / signal_0, 1)
    denom = jnp.where((starting_states != 0).astype(int),
                      output_diff / starting_states, 1)
    return jnp.absolute(jnp.divide(numer, denom))  # type: ignore
    # return jnp.divide(1, precision)


def compute_precision(starting_states, steady_states, signal_0, signal_1):
    signal_diff = signal_1 - signal_0
    output_diff = steady_states - starting_states

    return calculate_precision_core(output_diff, starting_states, signal_diff, signal_0)


def calculate_sensitivity_core(output_diff, starting_states, signal_diff, signal_0) -> jnp.ndarray:
    # denom = jnp.where(signal_0 != 0, signal_diff / signal_0, np.inf)
    numer = jnp.where((starting_states != 0).astype(int),
                      jnp.where(output_diff != 0, output_diff, np.nan) / starting_states, np.inf)

    return jnp.where(signal_0 != 0,
                     jnp.abs(jnp.divide(
                         numer, signal_diff / signal_0)),  # type: ignore
                     np.inf)  # type: ignore


def compute_sensitivity(signal_idx: int, starting_states, peaks):
    if signal_idx is None:
        return None
    signal_1 = peaks[signal_idx]
    signal_0 = starting_states[signal_idx]

    output_diff = peaks - starting_states
    signal_diff = signal_1 - signal_0

    return calculate_sensitivity_core(output_diff, starting_states, signal_diff, signal_0)


def compute_sensitivity2(starting_states, minv, maxv, signal_1, signal_0):
    output_diff = maxv - minv
    signal_diff = signal_1 - signal_0

    return calculate_sensitivity_core(output_diff, starting_states, signal_diff, signal_0)


def compute_sensitivity_simple(starting_states, peaks, signal_factor):
    numer = jnp.where((starting_states != 0).astype(int),
                      (peaks - starting_states) / starting_states, np.inf)
    return jnp.absolute(jnp.divide(
        numer, signal_factor))  # type: ignore


def compute_rmse(data: np.ndarray, ref_circuit_data: Optional[np.ndarray]):
    if ref_circuit_data is None:
        return jnp.zeros(shape=(jnp.shape(data)[0], 1))
    t_max = min([jnp.shape(data)[-1], jnp.shape(ref_circuit_data)[-1]])
    return jnp.sqrt(
        jnp.mean(jnp.power(data[:, :t_max] - ref_circuit_data[:, :t_max], 2), axis=1))


def compute_step_response_times(data, t, steady_states, signal_time: int):
    """ Compute the % settling time of the step response.
    Assumes that data starts pre-perturbation, but after an initial steady state 
    has been reached. """

    perc_settling_time = 0.01
    is_data_outside_stst = (data > (steady_states + steady_states * perc_settling_time)
                            ) | (data < (steady_states - steady_states * perc_settling_time))

    tpre = jnp.max(t * is_data_outside_stst, axis=-1).reshape(-1, 1)
    del is_data_outside_stst  # free up memory
    t_stable = t * (t > tpre)
    t_stable = jnp.where(t_stable != 0, t_stable, np.inf)
    tstop = jnp.min(t_stable, axis=-1)

    response_times = jnp.where(
        tstop != 0,
        tstop - signal_time,
        np.inf
    )
    return response_times


def frequency(data):
    spectrum = jnp.fft.fft(data)/len(data)
    spectrum = spectrum[range(int(len(data)/2))]
    freq = jnp.fft.fftfreq(len(spectrum))
    return freq


def log_distance(s, p):
    lin = np.array([np.logspace(6, -3, 2), np.logspace(-6, 3, 2)])
    return vec_distance(s, p, lin)


# Optimisation funcs
def mag(vec, **kwargs):
    return jnp.linalg.norm(vec, **kwargs)


def sp_prod(s, p, sp_factor=1, s_weight=0):
    """ Log product of s and p """
    s_lin = 1/p
    return s * (p * (s - s_lin))  # * sp_factor + s_weight)


def vec_distance(s, p, d):
    """ First row of each direction vector are the x's, second row are the y's """
    P = jnp.array([s, p]).T
    # P = [s.T, p.T]
    sp_rep = np.repeat(d[:, 0][:, None], repeats=len(s), axis=-1).T[:, :, None]
    AP = jnp.concatenate([sp_rep, P[:, :, None]], axis=-1)
    area = mag(jnp.cross(AP, d[None, :, :], axis=-1), axis=-1)
    D = area / mag(d)
    return D


def generate_base_analytics(data: jnp.ndarray, t: jnp.ndarray, labels: List[str],
                            signal_onehot: Optional[jnp.ndarray], signal_time,
                            ref_circuit_data: Optional[jnp.ndarray], include_deriv: bool = False) -> dict:
    """ Assuming [species, time] for data """
    signal_idxs = None if signal_onehot is None else [
        int(i) for i in (np.where(signal_onehot == 1)[0])]
    if data is None:
        return {}

    analytics = {
        # 'initial_steady_states': jnp.expand_dims(data[:, 0], axis=1),
        # 'max_amount': jnp.expand_dims(jnp.max(data, axis=1), axis=1),
        # 'min_amount': jnp.expand_dims(jnp.min(data, axis=1), axis=1),
        # 'steady_states': jnp.expand_dims(data[:, -1], axis=1)
    }

    analytics['initial_steady_states'] = data[:, 0]

    analytics['max_amount'] = jnp.max(data, axis=1)

    analytics['min_amount'] = jnp.min(data, axis=1)

    analytics['steady_states'] = data[:, -1]

    analytics['RMSE'] = compute_rmse(
        data, ref_circuit_data).squeeze()  # type: ignore

    first_derivative = compute_derivative(data)
    if include_deriv:
        analytics['first_derivative'] = first_derivative

    analytics['fold_change'] = compute_fold_change(
        starting_states=analytics['initial_steady_states'],
        steady_states=analytics['steady_states']
    )

    peaks = compute_peaks(initial_steady_states=analytics['initial_steady_states'], 
                          final_steady_states=analytics['steady_states'],
                          maxa=analytics['max_amount'], 
                          mina=analytics['min_amount'])

    analytics['overshoot'] = compute_overshoot(
        steady_states=analytics['steady_states'],
        peaks=peaks
    )

    if signal_idxs is not None:
        signal_labels = list(map(labels.__getitem__, signal_idxs))
        for s, s_idx in zip(signal_labels, signal_idxs):

            p_name = 'precision' if len(signal_idxs) == 1 else f'precision_wrt_species-{s_idx}'
            r_name = 'response_time' if len(signal_idxs) == 1 else f'response_time_wrt_species-{s_idx}'
            s_name = 'sensitivity' if len(signal_idxs) == 1 else f'sensitivity_wrt_species-{s_idx}'

            analytics[p_name] = compute_precision(
                starting_states=analytics['initial_steady_states'],
                steady_states=analytics['steady_states'],
                signal_0=analytics['initial_steady_states'][s_idx],
                signal_1=peaks[s_idx]
            )

            # t axis: 1
            t_end = np.min([len(t), data.shape[1]])
            analytics[r_name] = compute_step_response_times(
                data=data[:, :t_end], t=t[:t_end], 
                steady_states=analytics['steady_states'][:, None],
                signal_time=signal_time)

            analytics[s_name] = compute_sensitivity(
                signal_idx=s_idx, peaks=peaks, starting_states=analytics['initial_steady_states']
            )
    return analytics


def generate_differences_ratios(analytics: dict, ref_analytics) -> Tuple[dict, dict]:
    t_axis = 1
    differences = {}
    ratios = {}
    for k in ref_analytics.keys():
        if (ref_analytics[k].ndim > 1) and (analytics[k].ndim > 1):
            t_end = np.min([ref_analytics[k].shape[t_axis],
                           analytics[k].shape[t_axis]])
            differences[k +
                        DIFF_KEY] = jnp.subtract(analytics[k][:, :t_end], ref_analytics[k][:, :t_end])
            ratios[k + RATIO_KEY] = jnp.where(ref_analytics[k][:, :t_end] !=
                                              0, jnp.divide(analytics[k][:, :t_end], ref_analytics[k][:, :t_end]), np.inf)
        else:
            differences[k +
                        DIFF_KEY] = jnp.subtract(analytics[k], ref_analytics[k])
            ratios[k + RATIO_KEY] = jnp.where(ref_analytics[k] !=
                                              0, jnp.divide(analytics[k], ref_analytics[k]), np.inf)
    return differences, ratios


def generate_analytics(data: jnp.ndarray, time, labels: list, ref_circuit_data: Optional[jnp.ndarray] = None,
                       signal_onehot: Optional[jnp.ndarray] = None, signal_time=None):
    if data.shape[0] != len(labels):
        species_axis = data.shape.index(len(labels))
        data = np.swapaxes(data, 0, species_axis)  # type: ignore
    analytics = generate_base_analytics(data=data, t=time, labels=labels,
                                        signal_onehot=signal_onehot, signal_time=signal_time,
                                        ref_circuit_data=ref_circuit_data)

    # Differences & ratios
    # ref_analytics = generate_base_analytics(data=ref_circuit_data, t=time, labels=labels,
    #                                         signal_onehot=signal_onehot, signal_time=signal_time,
    #                                         ref_circuit_data=ref_circuit_data)
    # differences, ratios = generate_differences_ratios(analytics, ref_analytics)
    # return merge_dicts(analytics, differences, ratios)
    return analytics


class Timeseries():
    def __init__(self, data, time=None) -> None:
        self.data = data
        if data is None:
            self.time = None
        else:
            self.time = np.arange(np.shape(self.data)[
                                  1]) if time is None else time
