
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import List, Tuple, Union, Optional
from functools import partial
from datetime import datetime
import concurrent.futures
import inspect
import os
import gc
import logging
import diffrax as dfx
import numpy as np
import jax
import jax.numpy as jnp
from scipy import integrate

from bioreaction.model.data_containers import Species
from bioreaction.simulation.simfuncs.basic_de import bioreaction_sim, bioreaction_sim_expanded

from synbio_morpher.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from synbio_morpher.srv.parameter_prediction.interactions import InteractionSimulator, INTERACTION_FIELDS_TO_WRITE, MolecularInteractions
from synbio_morpher.utils.circuit.agnostic_circuits.circuit import Circuit, interactions_to_df
from synbio_morpher.utils.misc.helper import vanilla_return
from synbio_morpher.utils.misc.numerical import invert_onehot, zero_out_negs
from synbio_morpher.utils.misc.runtime import clear_caches
from synbio_morpher.utils.misc.type_handling import flatten_nested_dict, flatten_listlike, append_nest_dicts
from synbio_morpher.utils.results.visualisation import VisODE
from synbio_morpher.utils.modelling.deterministic import bioreaction_sim_dfx_expanded, bioreaction_sim_dfx_naive
from synbio_morpher.utils.modelling.solvers import get_diffrax_solver, make_stepsize_controller, simulate_steady_states
from synbio_morpher.utils.evolution.evolver import implement_mutation
from synbio_morpher.utils.evolution.mutation import Mutations
from synbio_morpher.utils.results.analytics.timeseries import generate_analytics
from synbio_morpher.utils.results.result_writer import ResultWriter


# Set modelling environment variables
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class CircuitModeller():

    def __init__(self, result_writer=None, config: dict = {}) -> None:
        self.result_writer = ResultWriter(
            None) if result_writer is None else result_writer
        self.steady_state_args = config['simulation_steady_state']
        self.simulator_args = config['interaction_simulator']
        self.simulation_args = config.get('simulation', {})
        self.interaction_simulator = InteractionSimulator(
            sim_args=self.simulator_args)
        self.interaction_factor = self.simulation_args.get(
            'interaction_factor', 1)
        self.discard_numerical_mutations = config['experiment'].get(
            'no_numerical', False)
        self.use_initial_to_add_signal = config.get(
            'simulation', {}).get('use_initial_to_add_signal', False)
        self.dt0 = config.get('simulation', {}).get('dt0', 0.3)
        self.dt1_factor = config.get('simulation', {}).get('dt1_factor', 5)
        self.dt1 = config.get('simulation', {}).get(
            'dt1', self.dt1_factor*self.dt0)
        self.t0 = config.get('simulation', {}).get('t0', 0)
        self.t1 = config.get('simulation', {}).get('t1', 10)
        self.threshold_steady_states = config.get('simulation', {}).get(
            'threshold_steady_states', 0.01)
        self.tmax = config.get('simulation', {}).get('tmax', self.t1)

        self.max_circuits = config.get('simulation', {}).get(
            'max_circuits', 10000)  # Maximum number of circuits to hold in memory
        self.debug_mode = config.get('experiment', {}).get('debug_mode', False)
        self.sim_func = None

        try:
            jax.config.update('jax_platform_name', config.get(
                'simulation', {}).get('device', 'gpu'))
        except:
            jax.config.update('jax_platform_name', 'cpu')
            logging.warning(
                'Could not set device to GPU, defaulting to CPU.')

        if self.debug_mode:
            jax.config.update("jax_disable_jit", True)
            if self.debug_mode == 1:
                os.environ["EQX_ON_ERROR"] = "breakpoint"

    def init_circuit(self, circuit: Circuit) -> Circuit:
        if self.simulation_args.get('use_rate_scaling', True):
            circuit = self.scale_rates([circuit])[0]
        circuit = self.find_steady_states([circuit])[0]
        return circuit

    def init_circuits(self, circuits: List[Circuit], batch=True) -> List[Circuit]:
        if self.simulation_args.get('use_rate_scaling', True):
            circuits = self.scale_rates(circuits)
        circuits = self.find_steady_states(circuits)
        return circuits

    def compute_interactions_core(self, circuit: Circuit) -> Circuit:

        if circuit.interactions_state == 'uninitialised' and (not self.debug_mode):
            if self.simulator_args['compute_by_filename'] and (circuit.subname == "ref_circuit") and (
                    circuit.data.source is not None) and os.path.exists(circuit.data.source):
                filename = circuit.data.source
            else:
                filename = None
            input_species = circuit.get_input_species()
            reactant_species = [
                r.species for r in circuit.qreactions.reactants]
            interactions = self.run_interaction_simulator(
                species=input_species,
                filename=filename,
                quantities=jnp.array([circuit.qreactions.reactants[reactant_species.index(s)].quantity for s in input_species]))
            circuit.interactions = interactions
            circuit.interactions_state = 'computed'
        elif circuit.interactions_state == 'loaded' or (circuit.interactions_state == 'computed'):
            pass
        elif self.debug_mode:
            circuit.init_interactions(init_dummy=True)
        else:
            circuit.init_interactions()

        return circuit

    # @time_it
    def compute_interactions(self, circuit: Circuit):
        circuit = self.compute_interactions_core(circuit)

        circuit.interactions.binding_rates_association = circuit.interactions.binding_rates_association * \
            self.interaction_factor
        circuit.interactions.binding_rates_dissociation = circuit.interactions.binding_rates_dissociation * \
            self.interaction_factor
        circuit.update_species_simulated_rates(circuit.interactions)
        return circuit

    def write_interactions(self, circuit: Circuit):
        for filename_addon in sorted(INTERACTION_FIELDS_TO_WRITE):
            interaction_matrix = circuit.interactions.__getattribute__(
                filename_addon)
            self.result_writer.output(
                data=interactions_to_df(
                    interaction_matrix,
                    labels=sorted([s.name for s in circuit.get_input_species()])),
                out_type='csv', out_name=circuit.name,
                overwrite=True, new_file=True,
                filename_addon=filename_addon, subfolder=filename_addon)
        return circuit

    def compute_interactions_batch(self, circuits: List[Circuit], batch=True):
        """ This will not work with GPU """
        # Make sure multi-threading is on
        if self.simulator_args['name'] == 'IntaRNA':
            if self.simulator_args['simulator_kwargs']['threads'] > 1:
                if self.simulator_args['simulator_kwargs']['raw_stdout'] == False:
                    logging.warning(
                        'For batching IntaRNA, setting raw_stdout to True, otherwise the output from the Python API will only return one interaction')
                    self.simulator_args['simulator_kwargs']['raw_stdout'] = True
        n_threads = self.simulator_args.get('multithread', 0)
        if n_threads > 0:
            for i in range(0, len(circuits), n_threads):
                j = i + n_threads if i + \
                    n_threads < len(circuits) else len(circuits)
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=j - i)
                results = list(executor.map(
                    self.compute_interactions, circuits[i:j]))
                circuits[i:j] = results

        else:
            if self.simulator_args['name'] == 'IntaRNA':
                logging.warning(
                    'For batch-computation of interaction strengths with IntaRNA, multi-threading is recommended. ' +
                    'Set `multithread` to > 0 in the config file within the simulator arguments (`interaction_simulator`) or ' +
                    '`threads` in the config file within the simulator arguments (`interaction_simulator`) for the kwargs (`simulator_kwargs`).')
            for i, c in enumerate(circuits):
                circuits[i] = self.compute_interactions(c)
        return circuits

    def run_interaction_simulator(self, species: List[Species], quantities: jnp.ndarray, filename=None) -> MolecularInteractions:
        data = {s: s.physical_data for s in species}
        return self.interaction_simulator.run(data, quantities=quantities, compute_by_filename=False)

    def find_steady_states(self, circuits: List[Circuit], batch=True) -> List[Circuit]:

        b_steady_states, t = self.compute_steady_states(circuits=circuits,
                                                        solver_type=self.steady_state_args['steady_state_solver'],
                                                        use_zero_rates=self.steady_state_args['use_zero_rates'])

        for circuit, steady_states in zip(circuits, b_steady_states):
            circuit.result_collector.add_result(
                data=steady_states,
                name='steady_states',
                category='time_series',
                vis_func=VisODE().plot,
                vis_kwargs={'t': t,
                            'legend': circuit.species_names,
                            'out_type': 'svg'},
                analytics_kwargs={'labels': [
                    s.name for s in circuit.model.species]},
                no_write=False)
        return circuits

    def compute_steady_states(self, circuits: List[Circuit],
                              solver_type: str = 'jax', use_zero_rates: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        t = np.zeros(0)
        if solver_type == 'ivp':
            b_copynumbers = []
            for circuit in circuits:
                r = circuit.qreactions.reactions
                if use_zero_rates and any((circuit.qreactions.reactions.reverse_rates - circuit.qreactions.reactions.forward_rates) > 1e2):
                    r = deepcopy(circuit.qreactions.reactions)
                    r.forward_rates = r.forward_rates * \
                        ((circuit.qreactions.reactions.reverse_rates -
                          circuit.qreactions.reactions.forward_rates) < 1e2) * 1

                signal_onehot = jnp.zeros_like(
                    circuit.signal.reactions_onehot) if circuit.use_prod_and_deg else jnp.zeros_like(circuit.signal.onehot)
                steady_state_result = integrate.solve_ivp(
                    partial(bioreaction_sim, args=None, reactions=r, signal=vanilla_return,
                            signal_onehot=signal_onehot),
                    (self.t0, self.t1),
                    y0=circuit.qreactions.quantities,
                    method=self.steady_state_args.get('method', 'Dopri5'))
                if not steady_state_result.success:
                    raise ValueError(
                        'Steady state could not be found through solve_ivp - possibly because units '
                        f'are in {circuit.interactions.units}. {SIMULATOR_UNITS}')
                copynumbers = steady_state_result.y
                b_copynumbers.append(copynumbers)
                t = steady_state_result.t

        elif solver_type == 'diffrax':
            ref_circuit = circuits[0]
            # Assuming all forward rates are the same
            forward_rates = ref_circuit.qreactions.reactions.forward_rates
            reverse_rates = np.asarray(
                [c.qreactions.reactions.reverse_rates for c in circuits])
            y0 = np.asarray([c.qreactions.quantities for c in circuits])
            signal_onehot = np.zeros_like(
                ref_circuit.signal.reactions_onehot) if ref_circuit.use_prod_and_deg else np.zeros_like(ref_circuit.signal.onehot)

            if self.debug_mode == 2:
                sim_func = partial(bioreaction_sim_dfx_naive,
                                   t0=self.t0, t1=self.t1, dt0=self.dt0,
                                   inputs=ref_circuit.qreactions.reactions.inputs,
                                   outputs=ref_circuit.qreactions.reactions.outputs,
                                   forward_rates=forward_rates,
                                   save_every_n_tsteps=5
                                   )
            else:
                sim_func = jax.vmap(partial(bioreaction_sim_dfx_expanded,
                                            t0=self.t0, t1=self.t1, dt0=self.dt0,
                                            signal=vanilla_return, signal_onehot=signal_onehot,
                                            inputs=ref_circuit.qreactions.reactions.inputs,
                                            outputs=ref_circuit.qreactions.reactions.outputs,
                                            forward_rates=forward_rates,
                                            solver=get_diffrax_solver(
                                                self.steady_state_args.get('method', 'Dopri5')),
                                            saveat=dfx.SaveAt(
                                                ts=np.linspace(self.t0, self.t1, int(np.min([200, self.t1-self.t0])))),
                                            stepsize_controller=make_stepsize_controller(self.t0, self.t1, self.dt0, self.dt1,
                                                                                         choice=self.steady_state_args.get('stepsize_controller', 'adaptive'))))

            b_copynumbers, t = simulate_steady_states(
                y0=y0, total_time=self.tmax, sim_func=sim_func,
                t0=self.t0, t1=self.t1,
                threshold=self.threshold_steady_states,
                reverse_rates=reverse_rates,
            )

            b_copynumbers = np.swapaxes(b_copynumbers, 1, 2)

        elif solver_type in ['torchode', 'torchdiffeq']:
            raise NotImplementedError()

        else:
            raise ValueError(
                f'The chosen solver `{solver_type}` is not supported.')

        return np.asarray(b_copynumbers), np.squeeze(t)

    def scale_rates(self, circuits: List[Circuit], batch: bool = True) -> List[Circuit]:
        forward_rates = np.zeros(
            (len(circuits), *circuits[0].qreactions.reactions.forward_rates.shape))
        reverse_rates = np.zeros(
            (len(circuits), *circuits[0].qreactions.reactions.forward_rates.shape))

        for i, c in enumerate(circuits):
            forward_rates[i] = np.array(c.qreactions.reactions.forward_rates)
            reverse_rates[i] = np.array(c.qreactions.reactions.reverse_rates)

        rate_max = np.max([np.max(np.asarray(forward_rates)),
                          np.max(np.asarray(reverse_rates))])

        # for i in range(len(circuits)):
        #     circuits[i].qreactions.reactions.forward_rates = circuits[i].qreactions.reactions.forward_rates / rate_max
        #     circuits[i].qreactions.reactions.reverse_rates = circuits[i].qreactions.reactions.reverse_rates / rate_max

        self.dt0 = np.min([1 / (5 * rate_max), self.dt0])
        self.dt1 = self.dt1_factor * self.dt0

        return circuits

    def simulate_signal_batch(self, circuits: List[Circuit],
                              ref_circuit: Circuit,
                              batch=True):

        self.prepare_internal_funcs(circuits)
        signal = ref_circuit.signal

        def prepare_batch_params(circuits: List[Circuit]):

            b_steady_states = np.zeros(
                (len(circuits), len(circuits[0].model.species)))
            b_og_states = np.array([c.result_collector.get_result(
                # type: ignore
                'steady_states').analytics['steady_states'].flatten(
            ) for i, c in enumerate(circuits)])
            b_reverse_rates = np.zeros(
                (len(circuits), *circuits[0].qreactions.reactions.reverse_rates.shape))

            add_sig_to_all_sigspecies = False
            onehots = signal.onehot
            if add_sig_to_all_sigspecies and (not self.use_initial_to_add_signal):
                species_chosen = circuits[0].model.species[np.argmax(
                    signal.onehot)]
                other_species = flatten_listlike(
                    [r.output for r in circuits[0].model.reactions if species_chosen in r.input])
                onehots_all_sigspecies = np.array([1 if s in other_species + [species_chosen]
                                                   else 0 for s in circuits[0].model.species])
                onehots = onehots_all_sigspecies
            for i, c in enumerate(circuits):
                """ Default """
                analytics_stst = c.result_collector.get_result(
                    'steady_states').analytics
                if analytics_stst is None:
                    raise ValueError(
                        f'Could not find analytics for steady state result.')
                if not c.use_prod_and_deg:
                    stst_key = 'initial_steady_states' if self.use_initial_to_add_signal else 'steady_states'
                    stst = analytics_stst[stst_key].flatten()
                    b_steady_states[i] = stst * ((onehots == 0) * 1) + \
                        (stst * signal.func.keywords['target']) * onehots
                else:
                    b_steady_states[i] = analytics_stst['steady_states'].flatten()
                b_reverse_rates[i] = c.qreactions.reactions.reverse_rates

            return b_steady_states, b_reverse_rates, b_og_states

        b_steady_states, b_reverse_rates, b_og_states = prepare_batch_params(
            circuits)

        s_time = datetime.now()
        b_new_copynumbers, t = simulate_steady_states(
            y0=b_steady_states, total_time=self.tmax, sim_func=self.sim_func,
            t0=self.t0, t1=self.t1,
            threshold=self.threshold_steady_states,
            reverse_rates=b_reverse_rates,
        )

        s_time = datetime.now() - s_time
        logging.warning(
            f'\t\tSimulating signal took {s_time.total_seconds()}s')

        if np.shape(b_new_copynumbers)[1] != ref_circuit.circuit_size and np.shape(b_new_copynumbers)[-1] == ref_circuit.circuit_size:
            b_new_copynumbers = np.swapaxes(b_new_copynumbers, 1, 2)

        # Fix first entry for signal species -> Warning: deletes final element in simulated data
        # We need to do this so that the precision and sensitivity calculations are correct
        for i, c in enumerate(circuits):
            if not c.use_prod_and_deg:
                b_new_copynumbers[i, :, :] = np.concatenate([np.expand_dims(
                    b_og_states[i, :], axis=1), b_new_copynumbers[i, :, :-1]], axis=1)

        # Get analytics batched too
        ref_circuits = [s for s in circuits if s.subname == 'ref_circuit']
        ref_idxs = [circuits.index(s) for s in ref_circuits]
        b_analytics_l = []

        # First check if the ref_circuit is leading
        if ref_circuit.name not in [c.name for c in ref_circuits]:
            ref_idxs.insert(0, None)  # type: ignore
        else:
            assert circuits.index(
                ref_circuit) == 0 or circuits[0].name == ref_circuit.name, f'The reference circuit should be leading or at idx 0, but is at idx {circuits.index(ref_circuit)}'

        a_time = datetime.now()

        ref_idxs2 = [len(circuits)] if len(
            ref_idxs) < 2 else ref_idxs[1:] + [len(circuits)]
        for ref_idx, ref_idx2 in zip(ref_idxs, ref_idxs2):

            if ref_idx is None:
                ref_circuit_result = ref_circuit.result_collector.get_result(
                    'signal')
                if ref_circuit_result is None:
                    raise ValueError(
                        'Reference circuit was not simulated and was not in this batch.')
                else:
                    ref_idx = 0
                    ref_circuit_data = ref_circuit_result.data
            else:
                ref_circuit_data = b_new_copynumbers[ref_idx]

            signal_time = signal.func.keywords['impulse_center'] if ref_circuit.use_prod_and_deg else t[1]

            analytics_func = jax.vmap(partial(
                generate_analytics, time=t, labels=[
                    s.name for s in ref_circuit.model.species],
                signal_onehot=signal.onehot, signal_time=signal_time,  # type: ignore
                ref_circuit_data=ref_circuit_data))  # type: ignore
            b_analytics = analytics_func(
                data=b_new_copynumbers[ref_idx:ref_idx2])
            b_analytics_l = append_nest_dicts(
                b_analytics_l, ref_idx2 - ref_idx, b_analytics)
        assert len(b_analytics_l) == len(
            circuits), f'There was a mismatch in length of analytics ({len(b_analytics_l)}) and circuits ({len(circuits)})'

        a_time = datetime.now() - a_time
        logging.warning(
            f'\t\tCalculating analytics took {a_time.total_seconds()}s')

        # Save for all circuits
        for i, (circuit, analytics) in enumerate(zip(circuits, b_analytics_l)):
            if self.discard_numerical_mutations and circuit.subname != 'ref_circuit':
                sig_data = None
                vis_func = lambda **kwargs: kwargs
            else:
                sig_data = b_new_copynumbers[i]
                vis_func = VisODE().plot

            circuits[i].result_collector.add_result(
                data=sig_data,
                name='signal',
                category='time_series',
                vis_func=vis_func,
                analytics=analytics,
                vis_kwargs={'t': t,
                            'legend': circuit.species_names,
                            'out_type': 'svg'})
        # return {top_name: {subname: circuits[len(v)*i + j] for j, subname in enumerate(v.keys())}
        #         for i, (top_name, v) in enumerate(.items())}
        clear_caches()
        return circuits

    def make_subcircuit(self, circuit: Circuit, mutation_name: str, mutation: Optional[Mutations] = None):

        subcircuit = deepcopy(circuit)
        subcircuit.reset_to_initial_state()
        subcircuit.strip_to_core()
        if mutation is None:
            mutation_from_circ = circuit.mutations[mutation_name]
            if type(mutation_from_circ) == Mutations:
                mutation = mutation_from_circ
            else:
                raise ValueError(f'Mutation not found for circuit {circuit}.')
        subcircuit.subname = mutation_name

        subcircuit = implement_mutation(circuit=subcircuit, mutation=mutation)
        return subcircuit

    def load_mutations(self, circuit: Circuit):
        subcircuits = [Circuit(config={}, as_mutation=True)
                       for m in flatten_nested_dict(circuit.mutations)]
        for i, (m_name, m) in enumerate(flatten_nested_dict(circuit.mutations).items()):
            if not m:
                continue
            subcircuits[i].subname = m_name

            # Can be by reference
            subcircuits[i].name = circuit.name
            subcircuits[i].circuit_size = circuit.circuit_size
            subcircuits[i].signal = circuit.signal
            subcircuits[i].use_prod_and_deg = circuit.use_prod_and_deg

            # Cannot be by ref
            subcircuits[i].model = deepcopy(circuit.model)
            subcircuits[i].species_names = circuit.species_names
            subcircuits[i].qreactions = deepcopy(circuit.qreactions)
            subcircuits[i] = implement_mutation(subcircuits[i], m)
        return subcircuits

    # @time_it
    def wrap_mutations(self, circuit: Circuit, methods: dict, include_normal_run=True,
                       write_to_subsystem=False):
        if write_to_subsystem:
            self.result_writer.subdivide_writing(circuit.name)
        mutation_dict = flatten_nested_dict(
            circuit.mutations)
        # logging.info(
        #     f'Running functions {methods} on circuit with {len(mutation_dict)} items.')

        self.result_writer.subdivide_writing(
            'mutations', safe_dir_change=False)
        for i, (name, mutation) in enumerate(mutation_dict.items()):
            # logging.info(f'Running methods on mutation {name} ({i})')
            if include_normal_run and i == 0:
                self.result_writer.unsubdivide_last_dir()
                circuit = self.apply_to_circuit(circuit, methods)
                self.result_writer.subdivide_writing(
                    'mutations', safe_dir_change=False)
            subcircuit = self.make_subcircuit(circuit, name, mutation)
            self.result_writer.subdivide_writing(name, safe_dir_change=False)
            self.apply_to_circuit(subcircuit, methods)
            self.result_writer.unsubdivide_last_dir()
        self.result_writer.unsubdivide()
        return circuit

    def prepare_internal_funcs(self, circuits: List[Circuit]):
        """ Create simulation function. If more customisation is needed per circuit, move
        variables into the relevant wrapper simulation method """

        # del self.sim_func

        ref_circuit = circuits[0]
        signal = ref_circuit.signal
        signal_f = vanilla_return if signal is None else signal.func
        signal_onehot = 1 if signal is None else signal.onehot

        if not all(signal == c.signal for c in circuits):
            logging.warning(
                f'Signal differs between circuits, but only first signal used for simulation.')

        if self.simulation_args['solver'] == 'diffrax':

            # Signal into parameter
            signal = ref_circuit.signal
            forward_rates = ref_circuit.qreactions.reactions.forward_rates * ((signal.reactions_onehot == 0) * 1) + \
                ref_circuit.qreactions.reactions.forward_rates * \
                signal.reactions_onehot * signal.func.keywords['target']

            self.sim_func = jax.jit(jax.vmap(
                partial(bioreaction_sim_dfx_expanded,
                        t0=self.t0, t1=self.t1, dt0=self.dt0,
                        signal=signal_f, signal_onehot=signal_onehot,
                        forward_rates=forward_rates,
                        inputs=ref_circuit.qreactions.reactions.inputs,
                        outputs=ref_circuit.qreactions.reactions.outputs,
                        solver=get_diffrax_solver(
                            self.simulation_args.get('method', 'Dopri5')),
                        saveat=dfx.SaveAt(
                            # t0=True, t1=True),
                            ts=np.linspace(self.t0, self.t1, 500)),  # int(np.min([500, self.t1-self.t0]))))
                        # ts=np.interp(np.logspace(0, 2, num=500), [1, np.power(10, 2)], [self.t0, self.t1])),  # Save more points early in the sim
                        stepsize_controller=make_stepsize_controller(self.t0, self.t1, self.dt0, self.dt1,
                                                                     choice=self.simulation_args.get('stepsize_controller', 'adaptive'))
                        )))

        elif self.simulation_args['solver'] == 'ivp':
            # way slower

            def b_ivp(y0: np.ndarray, forward_rates: np.ndarray, reverse_rates: np.ndarray):

                copynumbers = [None] * y0.shape[0]
                ts = [None] * y0.shape[0]
                for i, (y0i, forward_rates_i, reverse_rates_i) in enumerate(zip(y0, forward_rates, reverse_rates)):
                    signal_result = integrate.solve_ivp(
                        fun=partial(
                            bioreaction_sim_expanded,
                            args=None,
                            inputs=ref_circuit.qreactions.reactions.inputs,
                            outputs=ref_circuit.qreactions.reactions.outputs,
                            forward_rates=forward_rates_i, reverse_rates=reverse_rates_i
                        ),
                        t_span=(self.t0, self.t1),
                        y0=y0i,
                        method=self.steady_state_args.get('method', 'Dopri5')
                    )
                    if not signal_result.success:
                        raise ValueError(
                            'Signal could not be found through solve_ivp')
                    copynumbers[i] = signal_result.y
                    ts[i] = signal_result.t[-1]
                return copynumbers, ts

            self.sim_func = b_ivp
        else:
            raise ValueError(
                f'The simulation function could not be specified with solver option {self.simulation_args["solver"]}. Try `diffrax` or `ivp`.')

    def batch_circuits(self,
                       circuits: List[Circuit],
                       methods: dict,
                       batch_size: int = 0,
                       include_normal_run: bool = True,
                       write_to_subsystem=True,
                       cleanup=True):

        batch_size = len(circuits) if batch_size == 0 else batch_size
        n_circuits = len(circuits)

        num_subcircuits = [len(flatten_nested_dict(
            c.mutations)) + 1 for c in circuits]
        tot_subcircuits = sum(num_subcircuits)

        viable_circuit_nums = [0]
        next_viable = 0
        i = 0
        assert num_subcircuits[
            0] < self.max_circuits, f'The number of subcircuits {num_subcircuits[0]} in the first circuit is less than the max circuits specified ({self.max_circuits}).'
        while i < len(num_subcircuits):
            while (i < len(num_subcircuits)) and (next_viable + num_subcircuits[i] < self.max_circuits):
                next_viable += num_subcircuits[i]
                i += 1
            viable_circuit_nums.append(i)
            next_viable = 0

        logging.warning(
            f'\tFrom {n_circuits} circuits, a total of {tot_subcircuits} mutated circuits will be simulated.')

        start_time = datetime.now()
        for i, vi in enumerate(viable_circuit_nums[:-1]):
            single_batch_time = datetime.now()
            vf = min(viable_circuit_nums[i+1], n_circuits)
            logging.warning(
                f'\t\tStarting new round of viable circuits ({vi} - {vf} / {n_circuits})')

            # Preallocate then create subcircuits - otherwise memory leak
            subcircuits_time = datetime.now()
            subcircuits = list([np.nan] * sum(num_subcircuits[vi:vf]))
            c_idx = 0
            curr_subcircuits = []
            for i, circuit in enumerate(circuits[vi:vf]):
                curr_subcircuits = self.load_mutations(circuit)
                subcircuits[c_idx] = circuit
                subcircuits[c_idx+1:c_idx+1 +
                            len(curr_subcircuits)] = curr_subcircuits
                c_idx = c_idx + 1+len(curr_subcircuits)

            # if None in subcircuits:
            #     subcircuits = list(
            #         filter(lambda item: item is not None, subcircuits))
            if c_idx < len(subcircuits):
                subcircuits = subcircuits[:c_idx]
            subcircuits_time = datetime.now() - subcircuits_time
            logging.warning(
                f'\t\tMaking subcircuits {int(sum(num_subcircuits[:vi]))} - {int(sum(num_subcircuits[:vf]))} took {subcircuits_time.total_seconds()}s')

            # Batch
            gc.collect()
            ref_circuit = subcircuits[0]
            b_circuits = []
            for b in range(0, len(subcircuits), batch_size):
                logging.warning(
                    f'\tBatching {b} - {b+batch_size} circuits (out of {int(sum(num_subcircuits[:vi]))} - {int(sum(num_subcircuits[:vf]))} (total: {tot_subcircuits})) (Circuits: {vi} - {vf} of {n_circuits})')
                bf = b+batch_size if b + \
                    batch_size < len(subcircuits) else len(subcircuits)

                b_circuits = subcircuits[b:bf]
                if not b_circuits:
                    continue
                ref_circuit = self.run_batch(
                    b_circuits, methods, leading_ref_circuit=ref_circuit,
                    include_normal_run=include_normal_run,
                    write_to_subsystem=write_to_subsystem)

            single_batch_time = datetime.now() - single_batch_time
            logging.warning(
                f'Single batch: {single_batch_time} \nProjected time: {single_batch_time.total_seconds() * sum(num_subcircuits[vi:vf])/tot_subcircuits}s \nTotal time: {str(datetime.now() - start_time)}')
            if cleanup:
                del subcircuits
                del b_circuits
                del ref_circuit
                del curr_subcircuits
                del circuits[vi:vf]
            gc.collect()
        return circuits

    def run_batch(self,
                  subcircuits: List[Circuit],
                  methods: dict,
                  leading_ref_circuit: Circuit,
                  include_normal_run: bool = True,
                  write_to_subsystem: bool = True) -> Circuit:

        for method, kwargs in methods.items():
            method_time = datetime.now()
            ref_circuit = leading_ref_circuit
            if 'ref_circuit' in kwargs:
                kwargs.update({'ref_circuit': ref_circuit})
            logging.warning(
                f'\t\tRunning {len(subcircuits)} Subcircuits - {subcircuits[0].name}: {method}')

            if kwargs.get('batch'):  # method is batchable
                if hasattr(self, method):
                    if 'ref_circuit' in inspect.getfullargspec(getattr(self, method)).args:
                        kwargs.update({'ref_circuit': ref_circuit})
                    subcircuits = getattr(self, method)(subcircuits, **kwargs)
                else:
                    logging.warning(
                        f'Could not find method @{method} in class {self}')
            else:
                for i, subcircuit in enumerate(subcircuits):

                    dir_name = subcircuit.name if subcircuit.subname == 'ref_circuit' or not write_to_subsystem else os.path.join(
                        subcircuit.name, 'mutations', subcircuit.subname)

                    self.result_writer.subdivide_writing(
                        dir_name, safe_dir_change=True)

                    if subcircuit.subname == 'ref_circuit' and subcircuit.name != ref_circuit.name:
                        # ref_circuit.result_collector.delete_result('signal')
                        ref_circuit = subcircuit
                        if 'ref_circuit' in kwargs:
                            kwargs.update({'ref_circuit': ref_circuit})
                        if not include_normal_run:
                            continue

                    a_time = datetime.now()
                    subcircuit = self.apply_to_circuit(
                        subcircuit, {method: kwargs})
                    a_time = datetime.now() - a_time
                    subcircuits[i] = subcircuit
                self.result_writer.unsubdivide()

            method_time = datetime.now() - method_time
            logging.warning(
                f'\t\tMethod {method} took {method_time.total_seconds()}s')
        # Update the leading reference circuit to be the last ref circuti from this batch
        ref_circuits = [c for c in subcircuits if c.subname == 'ref_circuit']
        if ref_circuits:
            ref_circuit = ref_circuits[-1]
        else:
            ref_circuit = leading_ref_circuit
        del subcircuits
        return ref_circuit

    def apply_to_circuit(self, circuit: Circuit, methods: dict):
        for method, kwargs in methods.items():
            if hasattr(self, method):
                circuit = getattr(self, method)(circuit, **kwargs)
            else:
                logging.warning(
                    f'Could not find method @{method} in class {self}')
        return circuit

    def visualise_graph(self, circuit: Circuit, mode="pyvis", new_vis=False):
        self.result_writer.visualise_graph(circuit, mode, new_vis)

    def write_results(self, circuit: Circuit, new_report: bool = False, no_visualisations: bool = False,
                      only_numerical: bool = False, no_numerical: bool = False, no_analytics: bool = False):
        self.result_writer.write_all(
            circuit, new_report, no_visualisations=no_visualisations, only_numerical=only_numerical,
            no_numerical=no_numerical, no_analytics=no_analytics)
        return circuit
