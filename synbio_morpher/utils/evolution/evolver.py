
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import os
from typing import Tuple, List, Union, Optional
import numpy as np
import pandas as pd
from bioreaction.model.data_containers import Species
from synbio_morpher.srv.io.loaders.misc import load_csv
from synbio_morpher.utils.evolution.mutation import get_mutation_type_mapping, Mutations, EXCLUDED_NUCS
from synbio_morpher.utils.misc.type_handling import flatten_listlike
from synbio_morpher.utils.results.writer import DataWriter, kwargs_from_table


from synbio_morpher.utils.circuit.agnostic_circuits.circuit import Circuit


class Evolver():

    def __init__(self,
                 data_writer: DataWriter,
                 mutation_type: str = 'random',
                 sequence_type: str = 'RNA',
                 seed: int = 0,
                 concurrent_species_to_mutate: Union[str, list] = '') -> None:
        self.data_writer = data_writer
        self.mutation_type = mutation_type  # Not implemented
        self.out_name = 'mutations'
        self.out_type = 'csv'
        self.sequence_type = sequence_type
        self.mutation_type_mapping = get_mutation_type_mapping(sequence_type)
        self.concurrent_species_to_mutate = concurrent_species_to_mutate

        if seed is not None:
            np.random.seed(seed)

    def is_mutation_possible(self, circuit: Circuit) -> bool:
        if circuit.mutations_args['mutation_counts'] is None or circuit.mutations_args['mutation_nums_within_sequence'] is None:
            return False
        return True

    # def get_batch_mutator(self, algorithm: str):

    #     def mutation_sampler(sequence_length, num_mutations, batch_size=1):
    #         if sequence_length < num_mutations:
    #             logging.warning(
    #                 f'For sequences of length {sequence_length}, cannot mutate {num_mutations} times.')
    #         positions = np.random.choice(sequence_length, size=(
    #             num_mutations, batch_size), replace=False)
    #         return positions

    #     def sample_mutations(self, sequence: str, positions: list, mutation_nums_per_position: list) -> Tuple[list, list]:
    #         mutation_types = {}
    #         new_positions = []
    #         mutation_nums_per_position = mutation_nums_per_position * \
    #             len(positions) if len(
    #                 mutation_nums_per_position) == 1 else mutation_nums_per_position
    #         for p, n in zip(positions, mutation_nums_per_position):
    #             possible_transitions = self.mutation_type_mapping[sequence[p]]
    #             if n > len(possible_transitions):
    #                 logging.warning(
    #                     f'Cannot pick {n} when there are only {len(possible_transitions)} choices')
    #             mutation_types[p] = list(np.random.choice(
    #                 list(possible_transitions.values()), size=n, replace=False))
    #             new_positions.append([p] * n)

    #         return flatten_listlike(mutation_types.values()), flatten_listlike(new_positions)

    def mutate(self, circuit: Circuit, algorithm: str, write_to_subsystem=False):
        """ algorithm can be either random or all """
        if write_to_subsystem:
            self.data_writer.subdivide_writing(
                circuit.name, safe_dir_change=False)
        if self.is_mutation_possible(circuit):
            mutator = self.get_mutator(algorithm)
            circuit = mutator(circuit)
        else:
            logging.info('No mutation settings found, did not mutate.')

        if write_to_subsystem:
            self.data_writer.unsubdivide_last_dir()
        return circuit

    def get_mutator(self, algorithm):

        def mutation_sampler(sequence_length, num_mutations, batch_size=1):
            if sequence_length < num_mutations:
                logging.warning(
                    f'For sequences of length {sequence_length}, cannot mutate {num_mutations} times.')
            positions = np.random.choice(sequence_length, size=(
                num_mutations, batch_size), replace=False)
            return positions

        def make_species_iter(circuit: Circuit):
            spec_iter = circuit.model.species
            if self.concurrent_species_to_mutate:
                if type(self.concurrent_species_to_mutate) == str:
                    if self.concurrent_species_to_mutate == 'single_species_at_a_time':
                        spec_iter = circuit.model.species
                    elif self.concurrent_species_to_mutate in circuit.species_names:
                        spec_iter = [
                            s for s in circuit.model.species if s.name == self.concurrent_species_to_mutate]
                    else:
                        logging.warning(
                            f'Could not find {self.concurrent_species_to_mutate} among the circuit species:')
                        logging.warning(circuit.species_names)
                elif type(self.concurrent_species_to_mutate) == list:
                    spec_iter = []
                    for c in self.concurrent_species_to_mutate:
                        spec_iter.append([
                            s for s in circuit.model.species if s.name == c])
                    spec_iter = flatten_listlike(spec_iter)
                else:
                    logging.warning(
                        f'The mutation option {self.concurrent_species_to_mutate} is not implemented.')
            return spec_iter

        def rand_mutator(circuit: Circuit, algorithm: str, positions_chosen=None):
            spec_iter = make_species_iter(circuit)

            for specie in spec_iter:
                circuit.mutations[specie.name] = {}
                sequence = specie.physical_data
                if not sequence:
                    continue
                for mutation_nums_within_sequence in circuit.mutations_args['mutation_nums_within_sequence']:
                    for mutation_counts in circuit.mutations_args['mutation_counts']:
                        for mutation_idx in range(mutation_counts):

                            positions = mutation_sampler(
                                len(sequence), mutation_nums_within_sequence).flatten().tolist()
                            # positions = positions if positions_chosen is not None else mutation_sampler(
                            #     len(sequence), mutation_nums_within_sequence).flatten()
                            mutation_types, positions = self.sample_mutations(
                                sequence, positions, circuit.mutations_args['mutation_nums_per_position'])

                            mutation = self.make_mutations(
                                specie=specie, positions=positions,
                                mutation_idx=mutation_idx,
                                mutation_types=mutation_types, algorithm=algorithm,
                                template_file=circuit.data.source)

                            self.write_mutations(mutation)
                            circuit.mutations[specie.name][mutation.mutation_name] = mutation
            return circuit

        def all_mutator(circuit: Circuit, algorithm: str):

            def get_position_idxs(flat_idx: int, length: int, nesting: int):
                """ Returns index of size `nesting` corresponding to indexing a nesting-dimensional 
                array that is composed of np.arange(length), but repeated recursively. Such an 
                array grows exponentially in size and a 1-D version of it can simply be indexed
                multiple times instead. """
                return sorted(set([np.mod(int(np.divide(flat_idx, np.power(length, i))), length) for i in range(nesting)]))

            mutation_nums_per_position = [len(
                [k for k in self.mutation_type_mapping.keys() if k not in EXCLUDED_NUCS]) - 1]
            spec_iter = make_species_iter(circuit)
            for i, specie in enumerate(spec_iter):
                circuit.mutations[specie.name] = {}
                sequence = specie.physical_data
                sequence_l = len(sequence)
                if sequence_l == 0:
                    continue

                for mutation_nums_within_sequence in circuit.mutations_args['mutation_nums_within_sequence']:
                    total_mutations = np.power(
                        sequence_l, mutation_nums_within_sequence)

                    for r in range(total_mutations):
                        positions = get_position_idxs(
                            r, sequence_l, mutation_nums_within_sequence)
                        mutation_types, positions = self.sample_mutations(
                            sequence, positions, mutation_nums_per_position)
                        for i, (mt, p) in enumerate(zip(mutation_types, positions)):
                            mutation_idx = mutation_nums_per_position[0]*p + i
                            mutations = self.make_mutations(
                                specie=specie, positions=[
                                    p], mutation_idx=mutation_idx, mutation_types=[
                                    mt], algorithm=algorithm, template_file=circuit.data.source)

                            self.write_mutations(mutations)
                            circuit.mutations[specie.name][mutations.mutation_name] = mutations
            return circuit

        if algorithm == "random":
            return partial(rand_mutator, algorithm=algorithm)
        elif algorithm == "all":
            return partial(all_mutator, algorithm=algorithm)
        else:
            raise ValueError(
                f'Unrecognised mutation algorithm choice "{algorithm}"')

    def sample_mutations(self, sequence: str, positions: list, mutation_nums_per_position: list) -> Tuple[list, list]:
        mutation_types = {}
        new_positions = []
        mutation_nums_per_position = mutation_nums_per_position * \
            len(positions) if len(
                mutation_nums_per_position) == 1 else mutation_nums_per_position
        for p, n in zip(positions, mutation_nums_per_position):
            possible_transitions = self.mutation_type_mapping[sequence[p]]
            if n > len(possible_transitions):
                logging.warning(
                    f'Cannot pick {n} when there are only {len(possible_transitions)} choices')
            mutation_types[p] = np.random.choice(
                list(possible_transitions.values()), size=n, replace=False).tolist()
            new_positions.append([p] * n)

        return flatten_listlike(mutation_types.values()), flatten_listlike(new_positions)

    def make_mutations(self, specie: Species, positions: list,
                       mutation_idx: int, mutation_types: list,
                       algorithm: str, template_file: Optional[str]):
        mutation_count = len(positions)
        mutations = Mutations(
            mutation_name=specie.name+'_' +
            f'm{mutation_count}-' + str(
                mutation_idx),
            template_species=specie,
            template_name=specie.name,
            template_seq=specie.physical_data,
            mutation_types=mutation_types,
            count=mutation_count,
            positions=positions,
            sequence_type=self.sequence_type,
            template_file=template_file,
            algorithm=algorithm
        )
        return mutations

    def write_mutations(self, mutations: Mutations, overwrite=False):
        self.data_writer.output(
            out_type=self.out_type, out_name=self.out_name, data=mutations.as_table(), overwrite=overwrite)


def load_mutations(circuit, filename=None):
    table = load_csv(filename, load_as='pandas')
    if type(table) != pd.DataFrame:
        raise ValueError(f'Table from {filename} must return as pandas.')
    circuit.mutations = {}
    species_names = circuit.species_names
    for i in range(len(table)):
        mutating_species = circuit.model.species[species_names.index(
            table.iloc[i]['template_name'])]
        if mutating_species not in circuit.mutations:
            circuit.mutations[mutating_species] = {table['mutation_name'].iloc[i]: Mutations(
                template_species=mutating_species, **kwargs_from_table(Mutations, table=table.iloc[i]))}  # type: ignore
        else:
            circuit.mutations[mutating_species].update({table['mutation_name'].iloc[i]: Mutations(
                template_species=mutating_species, **kwargs_from_table(Mutations, table=table.iloc[i]))})  # type: ignore
        circuit.mutations[mutating_species][table.iloc[i]['mutation_name']].mutation_types = [int(
            v) for v in circuit.mutations[mutating_species][table.iloc[i]['mutation_name']].mutation_types]
        circuit.mutations[mutating_species][table.iloc[i]['mutation_name']].positions = [int(
            v) for v in circuit.mutations[mutating_species][table.iloc[i]['mutation_name']].positions]
    return circuit


def apply_mutation_to_sequence(sequence: str, mutation_positions: List[int], mutation_types: List[str]) -> str:
    sequence_arr = np.array([*sequence])
    sequence_arr[mutation_positions] = mutation_types
    return ''.join(sequence_arr)


def implement_mutation(circuit: Circuit, mutation: Mutations):

    if mutation.template_name in circuit.species_names:
        sidx = circuit.species_names.index(mutation.template_name)
        # circuit.model.species[sidx].name = mutation.mutation_name
        circuit.model.species[sidx].physical_data = mutation.get_sequence()
    else:
        raise KeyError(
            f'Could not find specie {mutation.template_name} in model for mutation {mutation.mutation_name}')

    circuit.qreactions.update_reactants(circuit.model)
    circuit.qreactions.update_reactions(circuit.model)
    return circuit
