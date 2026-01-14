
from bioreaction.simulation.manager import simulate_steady_states
from synbio_morpher.utils.results.analytics.timeseries import generate_analytics
from synbio_morpher.utils.misc.helper import vanilla_return
from synbio_morpher.utils.modelling.deterministic import bioreaction_sim_dfx_expanded
from synbio_morpher.srv.parameter_prediction.simulator import make_piecewise_stepcontrol
import numpy as np
import jax
import diffrax as dfx
from functools import partial

jax.config.update('jax_platform_name', 'cpu')


def scale_rates(forward_rates, reverse_rates, cushioning: int = 4):
    rate_max = np.max([np.max(np.asarray(forward_rates)),
                       np.max(np.asarray(reverse_rates))])

    dt0 = 1 / (cushioning * rate_max)
    return dt0


def mag(vec, **kwargs):
    return np.linalg.norm(vec, **kwargs)


def vec_distance(s, p, d):
    """ First row of each direction vector are the x's, second row are the y's """
    P = np.array([s, p]).T
    sp_rep = np.repeat(d[:, 0][:, None], repeats=len(s), axis=-1).T[:, :, None]
    AP = np.concatenate([sp_rep, P[:, :, None]], axis=-1)
    area = mag(np.cross(AP, d[None, :, :], axis=-1), axis=-1)
    D = area / mag(d)
    return D


def sp_prod(s, p, sp_factor=1, s_weight=0):
    return s * (p * sp_factor + s_weight)


def log_distance(s, p):
    lin = np.array([np.logspace(6, -3, 2), np.logspace(-6, 3, 2)])
    return vec_distance(s, p, lin)


def optimise_sp(s, p):
    return np.log(log_distance(s=s, p=p) * sp_prod(s, p, sp_factor=(p / s).max(), s_weight=np.log(p) / s))


def compute_analytics(y, t, labels, signal_onehot):
    y = np.swapaxes(y, 0, 1)

    analytics_func = partial(
        generate_analytics, time=t, labels=labels,
        signal_onehot=signal_onehot, signal_time=t[1],
        ref_circuit_data=None)
    return analytics_func(data=y, time=t, labels=labels)


def mini_sim(B11, B12, B13, B22, B23, B33):
    """ Only works for circuits with size n = 3 """
    unbound_species = ['RNA_0', 'RNA_1', 'RNA_2']
    species = ['RNA_0-0', 'RNA_0-1', 'RNA_0-2', 'RNA_1-1',
               'RNA_1-2', 'RNA_2-2', 'RNA_0', 'RNA_1', 'RNA_2']
    signal_species = ['RNA_0']
    output_species = ['RNA_1']
    s_idxs = [species.index(s) for s in signal_species]
    output_idxs = [species.index(s) for s in output_species]
    signal_onehot = np.array([1 if s in [species.index(
        ss) for ss in signal_species] else 0 for s in np.arange(len(species))])

    signal_target = 2
    k = 0.00150958097
    N0 = 200

    # Amounts
    y00 = np.array([[0, 0, 0, 0, 0, 0, N0, N0, N0]])

    # Reactions
    inputs = np.array([
        [0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 2],
    ])
    outputs = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
    ])

    # Rates
    reverse_rates = np.array([[B11, B12, B13, B22, B23, B33]])
    forward_rates = np.ones_like(reverse_rates) * k

    # Sim params
    t0 = 0
    t1 = 100
    # dt0 = scale_rates(forward_rates, reverse_rates, cushioning=4)
    dt0 = 0.0005555558569638981
    dt1_factor = 5
    dt1 = dt0 * dt1_factor
    max_steps = 16**4 * 10
    sim_func = jax.jit(partial(bioreaction_sim_dfx_expanded,
                               t0=t0, t1=t1, dt0=dt0,
                               signal=vanilla_return, signal_onehot=1,
                               forward_rates=forward_rates,
                               inputs=inputs,
                               outputs=outputs,
                               solver=dfx.Tsit5(),
                               saveat=dfx.SaveAt(
                                   ts=np.linspace(t0, t1, 500)),  # int(np.min([500, self.t1-self.t0]))))
                               max_steps=max_steps,
                               stepsize_controller=make_piecewise_stepcontrol(
                                   t0=t0, t1=t1, dt0=dt0, dt1=dt1)
                               ))

    y0, t = simulate_steady_states(y0=y00, total_time=t1-t0, sim_func=sim_func, t0=t0,
                                   t1=t1, threshold=0.05, reverse_rates=reverse_rates, disable_logging=True)
    y0 = np.array(y0.squeeze()[-1, :]).reshape(y00.shape)

    # Signal
    y0s = y0 * ((signal_onehot == 0) * 1) + y00 * signal_target * signal_onehot
    y, t = simulate_steady_states(y0s, total_time=t1-t0, sim_func=sim_func, t0=t0,
                                  t1=t1, threshold=0.05, reverse_rates=reverse_rates, disable_logging=True)
    y = np.concatenate([y0, y.squeeze()[:-1, :]], axis=0)

    analytics = compute_analytics(y, t, labels=np.arange(
        y.shape[-1]), signal_onehot=signal_onehot)

    s = analytics[f'sensitivity']
    p = analytics[f'precision']
    r = optimise_sp(
        s=s[:, None][tuple(output_idxs)], p=p[:, None][tuple(output_idxs)]
    )[0]

    return r, analytics, y, t
