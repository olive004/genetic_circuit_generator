
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
import inspect
import logging
import numpy as np
from typing import List
from synbio_morpher.utils.signal.signals_new import Signal
from synbio_morpher.utils.circuit.agnostic_circuits.circuit_manager import Circuit


# TODO: Delete
def get_signal_type(signal_type: str) -> type:
    return Signal
#     if signal_type == 'abstract':
#         return Signal
#     if signal_type == 'adaptation':
#         return AdaptationTarget
#     if signal_type == 'oscillation':
#         return OscillatingSignal
#     raise ValueError(f'Could not identify signal of type {signal_type}.')


def make_onehot_signals(circuit: Circuit, species_chosen: List[str]):
    """ Return an ndarray where 1 denotes 'signal input / output' 
    and 0 denotes the opposite"""
    onehot = np.zeros(circuit.circuit_size)
    idxs = [circuit.model.species.index(s) for s in circuit.model.species if s.name in species_chosen]
    onehot[idxs] = 1
    return onehot

def parse_sig_args(kwargs: dict, circuit: Circuit, SignalType = Signal):
    signal_init_args = inspect.getfullargspec(SignalType.__init__).args
    sig_kwargs = {}
    sig_kwargs['onehot'] = make_onehot_signals(circuit=circuit, species_chosen=kwargs["signal"]['inputs'])

    signal_species_idx = np.where(sig_kwargs['onehot'] == 1)[0]
    signal_species = [s for s in circuit.model.species if circuit.model.species.index(s) in signal_species_idx]
    reaction_has_signal = [bool(b) and all(b) for b in [[ro in signal_species for ro in r.output] for r in circuit.model.reactions]]
    reactions_onehot = np.ones_like(circuit.qreactions.reactions.forward_rates) * reaction_has_signal
    sig_kwargs['reactions_onehot'] = reactions_onehot

    sig_kwargs['time_interval'] = kwargs.get('simulation', {}).get('dt0', 1)
    for k, v in kwargs["signal"].items():
        if k in signal_init_args:
            sig_kwargs[k] = v
    if any([k not in sig_kwargs for k in signal_init_args if k != 'self']):
        logging.warning(f'Signal init kwargs missing from {sig_kwargs}')
    return sig_kwargs

