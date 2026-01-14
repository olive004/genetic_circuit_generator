import diffrax as dfx
from datetime import datetime
import numpy as np
import jax.numpy as jnp
from synbio_morpher.srv.parameter_prediction.simulator import make_piecewise_stepcontrol


def get_diffrax_solver(solver_type):

    solvers = {
        'Tsit5': dfx.Tsit5,
        'Dopri5': dfx.Dopri5,
        'Dopri8': dfx.Dopri8,
        'Heun': dfx.Heun,
        'Euler': dfx.Euler,
        'Midpoint': dfx.Midpoint,
        'Ralston': dfx.Ralston,
        'Bosh3': dfx.Bosh3,
        'ImplicitEuler': dfx.ImplicitEuler,
        'Kvaerno3': dfx.Kvaerno3,
        'Kvaerno4': dfx.Kvaerno4,
        'Kvaerno5': dfx.Kvaerno5,
        'Sil3': dfx.Sil3,
        'KenCarp3': dfx.KenCarp3,
        'KenCarp4': dfx.KenCarp4,
        'KenCarp5': dfx.KenCarp5,
        'SemiImplicitEuler': dfx.SemiImplicitEuler,
        'ReversibleHeun': dfx.ReversibleHeun,
        'LeapfrogMidpoint': dfx.LeapfrogMidpoint
    }

    assert solver_type in list(solvers.keys(
    )), f'Diffrax solver option {solver_type} not found. See https://docs.kidger.site/diffrax/api/solvers/ode_solvers/ for valid solvers.'

    return solvers[solver_type]()


def make_stepsize_controller(t0, t1, dt0, dt1, choice: str, **kwargs):
    """ The choice can be either log or piecewise """
    if choice == 'log':
        return make_log_stepcontrol(t0, t1, dt0, dt1, **kwargs)
    elif choice == 'piecewise':
        return make_piecewise_stepcontrol(t0=t0, t1=t1, dt0=dt0, dt1=dt1, **kwargs)
    elif choice == 'adaptive':
        return dfx.PIDController(rtol=1e-7, atol=1e-9)
    else:
        raise ValueError(
            f'The stepsize controller option `{choice}` is not available.')


def make_log_stepcontrol(t0, t1, dt0, dt1, upper_log: int = 3):
    num = 1000
    x = np.interp(np.logspace(0, upper_log, num=num), [
        1, np.power(10, upper_log)], [dt0, dt1])
    while np.cumsum(x)[-1] < t1:
        x = np.interp(np.logspace(0, upper_log, num=num), [
            1, np.power(10, upper_log)], [dt0, dt1])
        num += 1
    ts = np.cumsum(x)
    ts[0] = t0
    ts[-1] = t1
    return dfx.StepTo(ts)


def num_unsteadied(comparison, threshold):
    return np.sum(np.abs(comparison) > threshold)


def did_sim_break(y):
    if (np.sum(np.isnan(y)) > 0):
        raise ValueError(
            f'Simulation failed - some runs ({np.sum(np.isnan(y))/np.size(y) * 100} %) go to nan. Try lowering dt.')
    if (np.sum(y == np.inf) > 0):
        raise ValueError(
            f'Simulation failed - some runs ({np.sum(y == np.inf)/np.size(y) * 100} %) go to inf. Try lowering dt.')


def simulate_steady_states(y0, total_time, sim_func, t0, t1,
                           threshold=0.1, disable_logging=False,
                           **sim_kwargs):
    """ Simulate a function sim_func for a chunk of time in steps of t1 - t0, starting at 
    t0 and running until either the steady states have been reached (specified via threshold) 
    or until the total_time as has been reached. Assumes batching.

    Args:
    y0: initial state, shape = (batch, time, vars)
    t0: initial time
    t1: simulation chunk end time
    total_time: total time to run the simulation function over
    sim_kwargs: any (batchable) arguments left to give the simulation function,
        for example rates or other parameters. First arg must be y0
    threshold: minimum difference between the final states of two consecutive runs 
        for the state to be considered steady
    """

    ti = t0
    iter_time = datetime.now()
    # ys = y0
    # ys_full = ys
    # ts_full = 0
    while True:
        if ti == t0:
            y00 = y0
        else:
            y00 = ys[:, -1, :]

        ts, ys = sim_func(y00, **sim_kwargs)

        if np.sum(np.argmax(ts >= np.inf)) > 0:
            ys = ys[:, :np.argmax(ts >= np.inf), :]
            ts = ts[:, :np.argmax(ts >= np.inf)] + ti
        else:
            ys = ys
            ts = ts + ti

        did_sim_break(ys)

        if ti == t0:
            ys_full = ys
            ts_full = ts
        else:
            ys_full = np.concatenate([ys_full, ys], axis=1)
            ts_full = np.concatenate([ts_full, ts], axis=1)

        ti += t1 - t0

        if ys.shape[1] > 1:
            fderiv = jnp.gradient(ys[:, -5:, :], axis=1)[:, -1, :]
        else:
            fderiv = ys[:, -1, :] - y00
        if (num_unsteadied(fderiv, threshold) == 0) or (ti >= total_time):
            if not disable_logging:
                print('Done: ', datetime.now() - iter_time)
            break
        if not disable_logging:
            print('Steady states: ', ti, ' iterations. ', num_unsteadied(fderiv, threshold), ' left to steady out. ', datetime.now() - iter_time)

    if ts_full.ndim > 1:
        ts_full = ts_full[0]
    return np.array(ys_full), np.array(ts_full)