
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from bioreaction.model.data_tools import construct_model_fromnames
from bioreaction.model.data_containers import BasicModel
from synbio_morpher.utils.data.common import Data


def get_system_type(sys_type):
    if sys_type == "RNA":
        from synbio_morpher.utils.circuit.agnostic_circuits.circuit import Circuit
        return Circuit
    else:
        raise NotImplementedError(
            f"Desired system type {sys_type} not supported.")


def add_data_to_species(model: BasicModel, data: Data):
    for i, s in enumerate(model.species):
        if s.name in data.data.keys():
            model.species[i].physical_data = data.data[s.name]
    return model


def construct_bioreaction_model(data: Data, molecular_params: dict, include_prod_deg: bool):
    model = construct_model_fromnames(data.sample_names, include_prod_deg=include_prod_deg)
    model = add_data_to_species(model, data)
    for i in range(len(model.reactions)):
        model.reactions[i].forward_rate = 0
        model.reactions[i].reverse_rate = 0
        if model.reactions[i].input == [] and include_prod_deg:
            model.reactions[i].forward_rate = molecular_params['creation_rate']
        elif model.reactions[i].output == [] and include_prod_deg:
            model.reactions[i].forward_rate = molecular_params['degradation_rate']
    return model
