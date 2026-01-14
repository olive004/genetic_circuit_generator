

import os
import numpy as np


from synbio_morpher.srv.parameter_prediction.interactions import INTERACTION_FIELDS_TO_WRITE
from synbio_morpher.utils.common.setup import construct_circuit_from_cfg
from synbio_morpher.utils.data.data_format_tools.common import FORMAT_EXTS, load_json_as_dict
from synbio_morpher.utils.evolution.evolver import implement_mutation
from synbio_morpher.utils.misc.type_handling import inverse_dict, flatten_nested_dict
from synbio_morpher.utils.evolution.evolver import load_mutations


def load_results(top_circuit_dir, circuit):
    """ Load a circuit's results that have been simulated and written """
    
    numericals = [fn for fn in os.listdir(top_circuit_dir) if fn.endswith(inverse_dict(FORMAT_EXTS)['numpy']) or fn.endswith(inverse_dict(FORMAT_EXTS)['csv'])]
    
    for result_name in ['steady_states', 'signal']:
        bn = 'report_' + result_name + '.json'
        if os.path.isfile(os.path.join(top_circuit_dir, bn)):
            nfn = [n for n in numericals if result_name in n]
            nfn = nfn[0] if nfn else ''
            numerical = np.load(nfn) if os.path.isfile(nfn) else None
            analytics = load_json_as_dict(os.path.join(top_circuit_dir, bn))
            
            circuit.result_collector.add_result(
                data=numerical,
                name='steady_states',
                category='time_series',
                vis_func=None,
                analytics=analytics)

    return circuit


def load_circuit(top_circuit_dir, name, config, load_mutations_as_circuits = False):
    """ Load a circuit after it has been simulated and written """
    
    if 'interactions' not in config or config['interactions'] is None:
        config['interactions'] = {}
    for interaction in INTERACTION_FIELDS_TO_WRITE:
        if os.path.isdir(os.path.join(top_circuit_dir, interaction)):
            config["interactions"][interaction] = os.path.join(top_circuit_dir, interaction,
                os.listdir(os.path.join(top_circuit_dir, interaction))[0])

    circuit = construct_circuit_from_cfg(prev_configs=None, config_file=config) 
    circuit = load_results(top_circuit_dir, circuit)
    circuit.name = name
    
    if 'mutations.csv' in os.listdir(top_circuit_dir):
        circuit = load_mutations(circuit, filename=os.path.join(top_circuit_dir, 'mutations.csv'))
        
    if load_mutations_as_circuits and os.path.isdir(os.path.join(top_circuit_dir, 'mutations')):
        circuits = [circuit]
        for p, (m_name, m) in zip(
            sorted(os.listdir(os.path.join(top_circuit_dir, 'mutations'))),
            sorted(flatten_nested_dict(circuit.mutations).items())
        ): 
            assert p == m_name, f'Mismatch between mutation directory name and mutation name'
            c = load_circuit(
                os.path.join(top_circuit_dir, 'mutations', p), name=name, config=config, load_mutations_as_circuits=False)
            c.subname = m_name
            c = implement_mutation(c, m)
            circuits.append(c)
        return circuits
    else:
        return circuit
