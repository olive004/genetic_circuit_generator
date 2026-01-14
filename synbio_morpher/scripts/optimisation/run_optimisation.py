



import os
import sys
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import diffrax as dfx
from functools import partial
import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt

jax.config.update('jax_platform_name', 'cpu')


from synbio_morpher.utils.modelling.deterministic import bioreaction_sim_dfx_expanded
from synbio_morpher.utils.misc.helper import vanilla_return
from synbio_morpher.utils.results.analytics.timeseries import generate_analytics
from synbio_morpher.utils.results.analytics.naming import get_true_interaction_cols
from synbio_morpher.utils.results.experiments import Experiment, Protocol
from bioreaction.simulation.manager import simulate_steady_states
from synbio_morpher.srv.io.manage.script_manager import script_preamble


def scale_rates(forward_rates, reverse_rates, cushioning: int = 4):
    rate_max = np.max([np.max(np.asarray(forward_rates)),
                        np.max(np.asarray(reverse_rates))])

    dt0 = 1 / (cushioning * rate_max)
    return dt0


def optimise_sp(s, p):
    s_lin = 1 / p
    return s - s_lin


def compute_analytics(y, t, labels, signal_onehot):
    y = np.swapaxes(y, 0, 1)
    
    analytics_func = partial(
        generate_analytics, time=t, labels=labels,
        signal_onehot=signal_onehot, signal_time=0,
        ref_circuit_data=None)
    return analytics_func(data=y, time=t, labels=labels)


def R(B11, B12, B13, B22, B23, B33):
    unbound_species = ['RNA_0', 'RNA_1', 'RNA_2']
    species = ['RNA_0', 'RNA_1', 'RNA_2', 'RNA_0-0', 'RNA_0-1', 'RNA_0-2', 'RNA_1-1', 'RNA_1-2', 'RNA_2-2']
    signal_species = ['RNA_0']
    output_species = ['RNA_1']
    s_idxs = [species.index(s) for s in signal_species]
    output_idxs = [species.index(s) for s in output_species]
    signal_onehot = np.array([1 if s in [species.index(ss) for ss in signal_species] else 0 for s in np.arange(len(species))])
    
    signal_target = 2
    k = 0.00150958097
    N0 = 200
    
    # Amounts
    y00 = np.array([[N0, N0, N0, 0, 0, 0, 0, 0, 0]])
    
    # Reactions
    inputs = np.array([
        [2, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0],
    ])
    outputs = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ])
    
    # Rates
    reverse_rates = np.array([[B11, B12, B13, B22, B23, B33]])
    forward_rates = np.ones_like(reverse_rates) * k
    
    # Sim params
    t0 = 0
    t1 = 100
    dt0 = scale_rates(forward_rates, reverse_rates, cushioning=4)
    max_steps = 16**4 * 10
    # print('\n\nInput N:', y00)
    # print('Input B:', reverse_rates)
    sim_func = jax.jit(partial(bioreaction_sim_dfx_expanded,
        t0=t0, t1=t1, dt0=dt0,
        signal=vanilla_return, signal_onehot=1,
        forward_rates=forward_rates,
        inputs=inputs,
        outputs=outputs,
        solver=dfx.Tsit5(),
        saveat=dfx.SaveAt(
            ts=np.linspace(t0, t1, 500)),  # int(np.min([500, self.t1-self.t0]))))
        max_steps=max_steps
        ))
    
    y0, t = simulate_steady_states(y0=y00, total_time=t1-t0, sim_func=sim_func, t0=t0, t1=t1, threshold=0.1, reverse_rates=reverse_rates, disable_logging=True)
    y0 = np.array(y0.squeeze()[-1, :]).reshape(y00.shape)
    
    # Signal
    
    y0s = y0 * ((signal_onehot == 0) * 1) + y00 * signal_target * signal_onehot
    y, t = simulate_steady_states(y0s, total_time=t1-t0, sim_func=sim_func, t0=t0, t1=t1, threshold=0.1, reverse_rates=reverse_rates, disable_logging=True)
    y = np.concatenate([y0, y.squeeze()[:-1, :]], axis=0)
    y1 = np.array(y[-1, :])
        
    # print('Output:', y1)
    
    analytics = compute_analytics(y, t, labels=np.arange(y.shape[-1]), signal_onehot=signal_onehot)
    
    s = analytics['sensitivity']
    p = analytics['precision']
    # print(f'Sensitivity {output_idxs[0]}:', s[tuple(output_idxs)])
    # print(f'Precision {output_idxs[0]}:', p[tuple(output_idxs)])
    
    r = optimise_sp(
        s=s.squeeze()[tuple(output_idxs)], p=p.squeeze()[tuple(output_idxs)]
    )
    
    return r



def loglike(B):
    
    L = - 1 / (R(*B) + 0.0001)
    
    print(L)
    
    return L


def ptform(u):
    R_max = 1
    R_min = 0.00001
    x = R_max * u + R_min
    
    return x


def main(config=None, data_writer=None):
    
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "synbio_morpher", "scripts", "optimisation", "configs", "base_config.json"))

    def sample(ndim=6, maxiter: int = 1000):
        sampler = dynesty.NestedSampler(loglike, ptform, ndim)
        sampler.run_nested(maxiter=maxiter)
        sresults = sampler.results
        return sresults
    
    def visualise(sresults, data_writer):
        rfig, raxes = dyplot.runplot(sresults)
        tfig, taxes = dyplot.traceplot(sresults)
        cfig, caxes = dyplot.cornerplot(sresults)
        rfig.savefig(os.path.join(data_writer.top_write_dir, 'rfig.png'))
        tfig.savefig(os.path.join(data_writer.top_write_dir, 'tfig.png'))
        cfig.savefig(os.path.join(data_writer.top_write_dir, 'cfig.png'))
    
    protocols = [
        Protocol(
            partial(sample, maxiter=config['maxiter'])
        ),
        Protocol(
            partial(visualise, data_writer=data_writer), req_input=True
        )
    ]
    
    experiment = Experiment(config=config, config_file=config, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()
    
    return config, data_writer
    