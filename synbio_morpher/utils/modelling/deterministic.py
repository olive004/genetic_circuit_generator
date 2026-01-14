
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Union, Optional
import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx
from diffrax._step_size_controller.base import AbstractStepSizeController
from bioreaction.simulation.simfuncs.basic_de import bioreaction_sim, bioreaction_sim_expanded, one_step_de_sim_expanded
from bioreaction.model.data_containers import QuantifiedReactions
from synbio_morpher.utils.modelling.base import Modeller


class Deterministic(Modeller):
    def __init__(self, max_time=0, time_interval=1) -> None:
        super().__init__(max_time, time_interval)

    def dxdt_RNA(self, t, copynumbers, full_interactions, creation_rates, degradation_rates,
                 signal=None, signal_idx=None, identity_matrix=None):
        """ dx_dt = a + x * I * k_d * x' - x * ∂   for x=[A, B]
        x: the vector of copy numbers of the samples A, B, C...
        y: the vector of copy numbers of bound (nonfunctional) samples (AA, AB, AC...)
        a: the 'creation' rate, or for RNA samples, the transcription rate
        I: the identity matrix
        k_a: fixed binding rate of association between two RNA molecules (fixed at 1 or 1e6)
        k_d: a (symmetrical) matrix of interaction rates for the dissociation binding rate between each pair
            of samples - self-interactions are included
        ∂: the 'destruction' rate, or for RNA samples, the (uncoupled) degradation rate
        Data in format [sample, timestep] or [sample,]"""

        if signal_idx is not None:
            copynumbers[signal_idx] = signal

        xI = copynumbers * identity_matrix
        coupling = np.matmul(np.matmul(xI, full_interactions), copynumbers.T)
        dxdt = creation_rates - coupling.flatten() - \
            copynumbers.flatten() * degradation_rates

        return np.multiply(dxdt, self.time_interval)

    def dxdt_RNA_jnp(self, copynumbers, t, full_interactions, creation_rates, degradation_rates,
                     identity_matrix, signal=None, signal_idx=None):

        if signal_idx is not None:
            copynumbers = copynumbers.at[signal_idx].set(signal)

        xI = copynumbers * identity_matrix
        coupling = jnp.matmul(jnp.matmul(xI, full_interactions), copynumbers.T)

        dxdt = creation_rates - coupling - \
            copynumbers * degradation_rates
        new_copynumbers = copynumbers + dxdt

        if signal_idx is not None:
            new_copynumbers = new_copynumbers.at[signal_idx].set(signal)

        return jnp.where(new_copynumbers < 0, 0, new_copynumbers)


def simulate_signal_scan(copynumbers, time, full_interactions, creation_rates, degradation_rates,
                         identity_matrix, signal, signal_idx, one_step_func):
    def to_scan(carry, thingy):
        t, s = thingy
        return one_step_func(carry, t, full_interactions, creation_rates, degradation_rates,
                             identity_matrix, s, signal_idx), carry
    return jax.lax.scan(to_scan, copynumbers, (time, signal))


def bioreactions_simulate_signal_scan(copynumbers, time: np.ndarray, inputs, outputs, forward_rates, reverse_rates, signal, signal_onehot: np.ndarray):

    def to_scan(carry, thingy):
        t = thingy
        return bioreaction_sim_expanded(t, carry, args=(forward_rates, reverse_rates), inputs=inputs, outputs=outputs,
                                        # forward_rates=forward_rates,
                                        # reverse_rates=reverse_rates,
                                        # signal=signal,
                                        # signal_onehot=signal_onehot
                                        ), carry
    return jax.lax.scan(to_scan, copynumbers, (time))


def bioreaction_sim_wrapper(qreactions: QuantifiedReactions, t0, t1, dt0,
                            signal, signal_onehot: jnp.ndarray,
                            y0=None,
                            solver=dfx.Tsit5(),
                            saveat=dfx.SaveAt(t0=True, t1=True, steps=True)):
    """ The signal is a function that takes in t
    WARNING! Diffrax eqsolve will simulate to the end of the time series,
    then set time to infinity for the remainder until max_steps have been reached. """

    term = dfx.ODETerm(partial(bioreaction_sim, reactions=qreactions.reactions, signal=signal,
                               signal_onehot=signal_onehot))
    # y0 = qreactions.quantities if y0 is None else y0

    return dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0,
                           y0=y0, saveat=saveat, max_steps=16**4)


def bioreaction_sim_dfx_expanded(y0, t0, t1, dt0,
                                 inputs, outputs, forward_rates, reverse_rates,
                                 signal=None, signal_onehot: Optional[Union[int, np.ndarray]] = None,
                                 solver=dfx.Tsit5(),
                                 saveat=dfx.SaveAt(
                                     t0=True, t1=True, steps=True),
                                 max_steps=16**5,
                                 stepsize_controller: AbstractStepSizeController = dfx.ConstantStepSize(),
                                 return_as_sol=False):
    if type(stepsize_controller) == dfx.StepTo:
        dt0 = None
    term = dfx.ODETerm(
        partial(bioreaction_sim_expanded,
                inputs=inputs, outputs=outputs,
                # signal=signal,
                # signal_onehot=signal_onehot,
                forward_rates=forward_rates.squeeze(), reverse_rates=reverse_rates.squeeze()
                )
    )
    sol = dfx.diffeqsolve(term, solver,
                          t0=t0, t1=t1, dt0=dt0,
                          y0=y0.squeeze(),
                          saveat=saveat, max_steps=max_steps,
                          stepsize_controller=stepsize_controller)
    if return_as_sol:
        return sol
    return sol.ts, sol.ys


def bioreaction_sim_dfx_naive(y0: np.ndarray, reverse_rates,
                              t0, t1, dt0,
                              inputs, outputs, forward_rates,
                              save_every_n_tsteps: int = 1
                              ):
    """ Naive version of bioreaction_sim_dfx_expanded that does not use diffrax's ODE solver. """

    y = y0.copy()
    saves_t = np.arange(t0, t1, dt0*save_every_n_tsteps)
    num_saves = len(saves_t)
    saves_y = np.zeros((num_saves, *y0.shape))

    save_index = 0  # To keep track of saves
    for i, ti in enumerate(np.arange(t0, t1, dt0)):
        for iy, yi in enumerate(y):
            y[iy] = yi + one_step_de_sim_expanded(
                spec_conc=yi, inputs=inputs,
                outputs=outputs,
                forward_rates=forward_rates,
                reverse_rates=reverse_rates[iy]) * dt0

        if i % save_every_n_tsteps == 0:
            saves_y[save_index] = y
            save_index += 1

    return saves_t, saves_y
