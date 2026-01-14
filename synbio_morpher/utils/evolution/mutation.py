
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


from typing import List, Optional
from copy import deepcopy
from bioreaction.model.data_containers import Species
from synbio_morpher.utils.results.writer import Tabulated


EXCLUDED_NUCS = ['N']
mutation_type_mapping_RNA = {
    # Mutation idx from parent key to child key
    "A": {
        "C": 0,
        "G": 1,
        "U": 2
    },
    "C": {
        "A": 3,
        "G": 4,
        "U": 5
    },
    "G": {
        "A": 6,
        "C": 7,
        "U": 8
    },
    "U": {
        "A": 9,
        "C": 10,
        "G": 11
    },
    "N": {
        "A": 12,
        "C": 12,
        "G": 12,
        "U": 12
    }
}

mutation_type_mapping_DNA = {
    # Mutation idx from parent key to child key
    "A": {
        "C": 0,
        "G": 1,
        "T": 2
    },
    "C": {
        "A": 3,
        "G": 4,
        "T": 5
    },
    "G": {
        "A": 6,
        "C": 7,
        "T": 8
    },
    "T": {
        "A": 9,
        "C": 10,
        "G": 11
    }
}


def get_mutation_type_mapping(sequence_type):
    if sequence_type == 'RNA':
        return mutation_type_mapping_RNA
    elif sequence_type == 'DNA':
        return mutation_type_mapping_DNA
    else:
        raise ValueError(f'Unrecognised sequence type {sequence_type} provided - should be DNA or RNA.')


class Mutations(Tabulated):

    def __init__(self, mutation_name: str, template_species: Species,
                 template_name: str, template_seq: str,
                 template_file: Optional[str], positions: List[int], mutation_types: List[int],
                 count: int, sequence_type: str, algorithm: str) -> None:

        self.mutation_name = mutation_name
        self.template_species = template_species
        self.template_name = template_name # self.template_species.name
        self.template_seq = template_seq # self.template_species.sequence
        self.mutation_types = mutation_types
        self.positions = positions
        self.count = count
        self.algorithm = algorithm
        self.sequence_type = sequence_type
        self.template_file = template_file
        super().__init__()

    def get_table_properties(self):
        exempt_keys = ['template_species']
        props = self.__dict__
        for k in exempt_keys:
            props.pop(k)
        return list(props.keys()), list(props.values())

    def get_sequence(self):
        seq = list(deepcopy(self.template_seq))
        for i, p in enumerate(self.positions):
            seq[int(p)] = self.reverse_mut_mapping(int(self.mutation_types[i]))
        return ''.join(seq)

    def reverse_mut_mapping(self, mut_encoding: int):
        mutation_type_mapping = get_mutation_type_mapping(self.sequence_type)
        for k, v in mutation_type_mapping.items():
            if mut_encoding in list(v.values()):
                for mut, enc in v.items():
                    if enc == mut_encoding:
                        return mut
        raise ValueError(
            f'Could not find mutation for mapping key {mut_encoding}.')


def reverse_mut_mapping(mut_encoding: int, sequence_type: str = 'RNA'):
    for k, v in get_mutation_type_mapping(sequence_type).items():
        if mut_encoding in list(v.values()):
            for mut, enc in v.items():
                if enc == mut_encoding:
                    return mut
    raise ValueError(
        f'Could not find mutation for mapping key {mut_encoding}.')

