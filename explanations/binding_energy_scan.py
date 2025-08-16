from bioreaction.model.data_tools import construct_model_fromnames
from bioreaction.simulation.simfuncs.basic_de import bioreaction_sim_expanded
from bioreaction.simulation.basic_sim import convert_model, BasicSimParams, BasicSimState
from bioreaction.simulation.manager import simulate_steady_states
from bioreaction.simulation.med_sim import MedSimParams

from datetime import datetime
import diffrax as dfx
from tqdm import tqdm
import jax
import pandas as pd
import scipy
import jax.numpy as jnp
import numpy as np
from copy import deepcopy
from functools import partial
import os
import sys

import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.style.use('seaborn-v0_8')

os.environ["TF_CPP_MIN_LOG_LOVEL"] = "0"
jax.config.update('jax_platform_name', 'gpu')
jax.devices()

if __package__ is None or (__package__ == ''):

    module_path = os.path.abspath(os.path.join('..'))
    sys.path.append(module_path)
    sys.path.append(os.path.abspath(os.path.join('.')))

    __package__ = os.path.basename(module_path)


from synbio_morpher.utils.results.analytics.timeseries import compute_sensitivity_simple, get_precision_simp, get_step_response_times, get_overshoot, get_peaks
from synbio_morpher.utils.misc.units import per_mol_to_per_molecule
from synbio_morpher.utils.misc.type_handling import flatten_listlike
from synbio_morpher.utils.misc.numerical import make_symmetrical_matrix_from_sequence
from synbio_morpher.utils.modelling.deterministic import bioreaction_sim_dfx_expanded


def scale_rates(sim_model, reverse_rates):
    m = np.max([sim_model.forward_rates.max(), reverse_rates.max()])
    sim_model.forward_rates = sim_model.forward_rates/m
    return sim_model, m


def update_model_rates(model, a=None, d=None, ka=None, kd=None):
    for i, r in enumerate(model.reactions):
        if not r.input:  # 0 -> RNA
            if a is not None:
                model.reactions[i].forward_rate = a[model.species.index(
                    r.output[0])]
                model.reactions[i].reverse_rate = 0
        elif not r.output:  # RNA -> 0
            if d is not None:
                model.reactions[i].forward_rate = d[model.species.index(
                    r.input[0])]
                model.reactions[i].reverse_rate = 0
        else:
            if ka is not None:
                model.reactions[i].forward_rate = ka[model.species.index(r.input[0]),
                                                     model.species.index(r.input[1])]
            if kd is not None:
                model.reactions[i].reverse_rate = kd[model.species.index(r.input[0]),
                                                     model.species.index(r.input[1])]
    return model


def define_matrices(num_species, size_interaction_array, num_unique_interactions, num_analytic_types):
    matrix_dimensions = tuple(
        [num_species] + [size_interaction_array]*num_unique_interactions)
    matrix_size = num_species * \
        np.power(size_interaction_array, num_unique_interactions)
    assert matrix_size == np.prod(list(
        matrix_dimensions)), 'Something is off about the intended size of the matrix'

    all_analytic_matrices = []
    for _analytic in range(num_analytic_types):
        all_analytic_matrices.append(np.zeros(
            matrix_dimensions, dtype=np.float32))
    return all_analytic_matrices


def make_batch_rates(model, rates, interaction_matrices):
    index_translation = []
    for i, r in enumerate(model.reactions):
        if r.output and r.input:
            index_translation.append([i, model.species.index(r.input[0]),
                                      model.species.index(r.input[1])])
    rates = np.repeat(
        np.expand_dims(rates, axis=0),
        interaction_matrices.shape[0], axis=0
    )
    for i, k, j in index_translation:
        rates[:, i] = interaction_matrices[:, k, j]
    return rates


def clear_gpu():
    backend = jax.lib.xla_bridge.get_backend()
    for buf in backend.live_buffers():
        buf.delete()


def get_loop_simfunc(params, sim_model, saveat):
    def to_vmap_basic(starting_state, reverse_rates):
        return partial(bioreaction_sim_dfx_expanded,
                       t0=params.t_start, t1=params.t_end, dt0=params.delta_t,
                       signal=None, signal_onehot=None,
                       forward_rates=sim_model.forward_rates,
                       inputs=sim_model.inputs,
                       outputs=sim_model.outputs,
                       solver=dfx.Tsit5(),
                       saveat=saveat,
                       max_steps=int(np.min(
                           [np.max([16**4, ((params.t_end - params.t_start)/params.delta_t) * 5]), 16**6]))
                       )(y0=starting_state, reverse_rates=reverse_rates)

    return jax.jit(jax.vmap(to_vmap_basic))


def num_unsteadied(comparison):
    return np.sum(np.abs(comparison) > 0.1)


def get_full_steady_states(y0, total_time, reverse_rates, sim_model, params, saveat=dfx.SaveAt(t0=True, t1=True, steps=True)):

    sim_func = get_loop_simfunc(
        params, sim_model=sim_model, saveat=saveat)
    ti = params.t_start
    iter_time = datetime.now()
    while True:
        if ti == params.t_start:
            y00 = y0
        else:
            y00 = ys[:, -1, :]

        x_res = sim_func(y00, reverse_rates)

        if np.sum(np.argmax(x_res.ts >= np.inf)) > 0:
            ys = x_res.ys[:, :np.argmax(x_res.ts >= np.inf), :]
            ts = x_res.ts[:, :np.argmax(x_res.ts >= np.inf)] + ti
        else:
            ys = x_res.ys
            ts = x_res.ts + ti

        if ti == params.t_start:
            ys_full = ys
            ts_full = ts
        else:
            ys_full = np.concatenate([ys_full, ys], axis=1)
            ts_full = np.concatenate([ts_full, ts], axis=1)

        if (num_unsteadied(ys[:, -1, :] - y00) == 0) or (ti >= total_time):
            print('Done: ', datetime.now() - iter_time)
            break
        print('Steady states: ', ti, ' iterations. ', (num_unsteadied(
            ys[:, -1, :] - y00)), ' left to steady out. ', datetime.now() - iter_time)

        ti += params.t_end - params.t_start

    return np.array(ys_full), np.array(ts_full[0])


def get_analytics(steady_states, final_states, t, K_eqs, model,
                  species_names, species_types, input_species_idx,
                  starting_circuit_idx, signal_factor):
    batchsize = steady_states.shape[0]

    peaks = get_peaks(steady_states, final_states[:, -1, :],
                      final_states.max(axis=1), final_states.min(axis=1)) 

    precision = np.array(jax.jit(jax.vmap(partial(get_precision_simp, signal_factor=signal_factor)))(
        starting_states=steady_states,
        steady_states=final_states[:, -1, :]
    ))
    sensitivity = np.array(jax.jit(jax.vmap(partial(compute_sensitivity_simple, signal_factor=signal_factor)))(
        peaks=peaks, starting_states=steady_states
    ))
    overshoots = np.array(get_overshoot(np.swapaxes(
        final_states, 1, 2)[:, :, -1], peaks=peaks))
    clear_gpu()
    resp_vmap = jax.jit(jax.vmap(partial(get_step_response_times,
                                         t=t, signal_time=0.0, signal_idx=input_species_idx)))  # np.expand_dims(t, 0)
    response_times = np.array(resp_vmap(
        deriv=np.gradient(np.swapaxes(final_states, 1, 2), axis=-1),
        data=np.swapaxes(final_states, 1, 2), steady_states=np.expand_dims(
            np.swapaxes(final_states, 1, 2)[:, :, -1], axis=-1)))

    analytics = {
        'precision': precision.flatten(),
        'sensitivity': sensitivity.flatten(),
        'response_times': response_times.flatten(),
        'overshoots': overshoots.flatten()
    }
    analytics_df = pd.DataFrame.from_dict(analytics)
    analytics_df['circuit_name'] = np.repeat(np.expand_dims(
        np.arange(starting_circuit_idx, starting_circuit_idx+batchsize), axis=0), len(model.species), axis=1)[0]
    analytics_df['relative_circ_strength'] = np.repeat(np.expand_dims(
        [K_eqs[i].mean() for i in range(batchsize)], axis=0), len(model.species), axis=1)[0]
    analytics_df['sample_name'] = flatten_listlike(
        [species_names for i in range(batchsize)])
    analytics_df['species_type'] = flatten_listlike(
        [species_types for i in range(batchsize)])
    return analytics_df


def plot_scan(final_states, t_final, bi, species_names_onlyin, num_show_species, num_show_species_i=0):
    max_per_plot = np.min([final_states.shape[0], 49])
    num_plots = np.max([1, int(np.ceil(final_states.shape[0] / max_per_plot))])
    for p in tqdm(range(num_plots)):
        num_rows = int(np.sqrt(max_per_plot))
        plt.figure(figsize=(50, 50))
        for i in range(num_rows**2):
            ii = (p*(num_rows**2))+i
            if ii >= final_states.shape[0]:
                break
            ax = plt.subplot(num_rows, num_rows, i+1)
            plt.plot(t_final, final_states[ii, :,
                     num_show_species_i:num_show_species])
            plt.title(str(bi+ii))
        plt.legend(species_names_onlyin)
        plt.savefig(os.path.join('output', '5_Keqs_exp',
                    f'final_states_{(bi+p)*max_per_plot}-{(bi+p)*max_per_plot+max_per_plot}_{num_show_species_i}{num_show_species}.svg'))
        plt.close()


def adjust_sim_params():
    dt = 0.1

    params_steady = MedSimParams(t_start=0, t_end=1000.0, delta_t=dt,
                                 poisson_sim_reactions=None, brownian_sim_reaction=None)
    total_t_steady = 3000000
    params_final = MedSimParams(t_start=0, t_end=1000.0, delta_t=dt,
                                poisson_sim_reactions=None, brownian_sim_reaction=None)
    total_t_final = 3000000

    return params_steady, total_t_steady, params_final, total_t_final


def scan_all_params(s0, K_eqs, b_reverse_rates, model, include_prod_deg=True):
    batchsize = int(b_reverse_rates.shape[0] / 1)
    sig_factor = 2
    a_sig = a * 1
    a_sig[input_species_idx] = a[input_species_idx] * sig_factor
    model_steady = update_model_rates(model, a=a)
    model_sig = update_model_rates(deepcopy(model), a=a_sig)

    species_names = [s.name for s in model.species]
    species_names_onlyin = species_names[:num_species]
    species_types = ['unbound' if s in model.species[:num_species]
                     else 'bound' for s in model.species]

    analytics_df = pd.DataFrame()
    params_steady, total_t_steady, params_final, total_t_final = adjust_sim_params()

    for bi in range(0, b_reverse_rates.shape[0], batchsize):
        clear_gpu()
        bf = min(bi+batchsize, b_reverse_rates.shape[0])
        reverse_rates = b_reverse_rates[bi:bf]
        s0_i = np.repeat(np.expand_dims(s0, axis=0),
                         repeats=reverse_rates.shape[0], axis=0)

        sim_model, scaling = scale_rates(
            convert_model(model_steady), reverse_rates)
        # x_steady, t_steady = get_full_steady_states(
        #     s0_i, total_time=total_t_steady,
        #     reverse_rates=reverse_rates/scaling, sim_model=sim_model,
        #     params=params_steady, saveat=dfx.SaveAt(t1=True))
        x_steady, t_steady = simulate_steady_states(
            s0_i, total_time=total_t_steady,
            sim_func=get_loop_simfunc(
                params_steady, sim_model=sim_model, 
                saveat=dfx.SaveAt(t1=True)),
                # saveat=dfx.SaveAt(ts=np.linspace(params_final.t_start, params_final.t_end, 100))),
            t0=params_steady.t_start,
            t1=params_steady.t_end, threshold=0.2, reverse_rates=reverse_rates/scaling)

        # clear_gpu()
        sim_model, scaling = scale_rates(
            convert_model(model_sig), reverse_rates)
        if not include_prod_deg:
            sig_onehot = (np.arange(len(model.species))
                          == input_species_idx) * 1
            # x_steady_sig = x_steady[:, -1, :] * ((sig_onehot == 0) * 1) + \
            #     x_steady[:, -1, :] * sig_onehot * sig_factor
            x_steady_sig = x_steady[:, -1, :] + \
                s0_i * (sig_factor - 1) * sig_onehot
        else:
            x_steady_sig = x_steady[:, -1, :]
        # x_final, t_final = get_full_steady_states(
        #     x_steady_sig, total_time=total_t_final, reverse_rates=reverse_rates/scaling,
        #     sim_model=sim_model, params=params_final,
        #     saveat=dfx.SaveAt(ts=np.linspace(params_final.t_start, params_final.t_end, 1000)))
        x_final, t_final = simulate_steady_states(
            x_steady_sig, total_time=total_t_final,
            sim_func=get_loop_simfunc(params_final, sim_model=sim_model, saveat=dfx.SaveAt(
                ts=np.linspace(params_final.t_start, params_final.t_end, 100))),
            t0=params_final.t_start,
            t1=params_final.t_end, threshold=0.05, reverse_rates=reverse_rates/scaling)

        clear_gpu()
        # x_steady_sig[:, input_species_idx] = s0_i[:, input_species_idx]
        analytics = get_analytics(
            x_steady[:, -1, :], x_final, t_final, K_eqs[bi:bf], model,
            species_names, species_types, input_species_idx, bi, signal_factor=sig_factor - 1)
        analytics_df = pd.concat([analytics_df, analytics], axis=0)
        analytics_df.to_csv(os.path.join('output', '5_Keqs_exp', 'df.csv'))

        np.save(os.path.join('output', '5_Keqs_exp', f't_{bi}.npy'), t_final)
        np.save(os.path.join('output', '5_Keqs_exp', f'final_states_{bi}-{bf}.npy'),
                x_final[bi:bf, :, :])

        plot_scan(x_final, t_final, bi, species_names_onlyin, num_species)
        plot_scan(x_final, t_final, bi, species_names, len(species_names))
        plot_scan(x_final, t_final, bi,
                  species_names_onlyin[1:num_species], num_species, 1)
    return analytics_df


if __name__ == '__main__':
    num_species = 3
    input_species_idx = 0
    output_species_idx = 1
    include_prod_deg = False
    model = construct_model_fromnames(
        [str(i) for i in range(num_species)], include_prod_deg=include_prod_deg)
    model.species = model.species[-num_species:] + model.species[:-num_species]

    Keq = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]
    )
    # From synbio_morpher/utils/common/configs/RNA_circuit/molecular_params.json
    a = np.ones(num_species) * 0.08333
    # a[1] = a[1] * 1.2
    # a[2] = a[2] * 0.8
    d = np.ones(len(model.species)) * 0.0008333
    # d = np.ones(num_species) * 0.0008333
    # d[0] = d[0] * 0.5
    # d[1] = d[1] * 1
    # d[2] = d[2] * 1
    ka = np.ones_like(Keq) * per_mol_to_per_molecule(1000000)
    kd = ka/Keq

    model = update_model_rates(model, a, d, ka, kd)

    s0_s = 100
    s0 = np.concatenate([
        np.ones(num_species) * s0_s,
        np.zeros(len(model.species) - num_species)])
    # s0[1] = 170
    # s0[2] = 55
    starting_state = BasicSimState(concentrations=s0)

    # K_eqs_range = np.array([0.02, 0.8, 1.2, 4, 30, 60, 110])
    K_eqs_range = np.array([0.0001, 0.4, 1.0, 5, 40])
    # K_eqs_range = np.array([0.01, 0.5, 1.0, 5, 40])
    num_Keqs = np.size(K_eqs_range)
    num_unique_interactions = np.math.factorial(num_species)

    # Set loop vars
    analytic_types = ['precision']
    all_analytic_matrices = define_matrices(
        num_species, num_Keqs, num_unique_interactions, len(analytic_types))

    total_iterations = np.power(num_Keqs, num_unique_interactions)
    total_processes = 1
    sub_process = 0
    num_iterations = int(total_iterations / total_processes)

    print('Making ', num_iterations,
          ' parameter combinations for equilibrium constants range: ', K_eqs_range)

    def make_keqs():
        K_eqs = np.zeros((num_iterations, num_species, num_species))
        pows = np.power(num_Keqs, np.arange(num_unique_interactions))
        for i in range(num_iterations):
            interaction_strength_choices = np.floor(
                np.mod(i / pows, num_Keqs)).astype(int)
            if interaction_strength_choices[1] < interaction_strength_choices[3]:
                continue
            flat_triangle = K_eqs_range[list(
                interaction_strength_choices)]
            K_eqs[i] = make_symmetrical_matrix_from_sequence(
                flat_triangle, num_species)
        return K_eqs

    K_eqs = jax.jit(make_keqs, backend='cpu')()
    didx = np.flatnonzero((K_eqs == 0).all((1, 2)))
    K_eqs = np.delete(K_eqs, didx, axis=0)

    print(K_eqs.shape[0], 'unique combinations to scan.')

    kds = ka / K_eqs
    sim_model = convert_model(model)
    b_reverse_rates = make_batch_rates(model, sim_model.reverse_rates, kds)

    clear_gpu()
    batchsize = len(K_eqs)
    analytics_df = scan_all_params(
        s0, K_eqs[:batchsize], b_reverse_rates[:batchsize], model, include_prod_deg)
    analytics_df.to_csv(os.path.join('output', '5_Keqs_exp', 'df.csv'))
