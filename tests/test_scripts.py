

# import logging
# import os
# import unittest
# import inspect

# import numpy as np
# from synbio_morpher.srv.io.manage.sys_interface import make_filename_safely
# from synbio_morpher.utils.results.result_writer import ResultWriter
# from synbio_morpher.utils.common.setup import ESSENTIAL_KWARGS
# from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
# from synbio_morpher.utils.misc.io import convert_pathname_to_module, get_pathnames, get_subdirectories
# from synbio_morpher.utils.misc.numerical import make_symmetrical_matrix_from_sequence, triangular_sequence
# from synbio_morpher.utils.common.setup import construct_circuit_from_cfg, prepare_config
# from synbio_morpher.utils.circuit.agnostic_circuits.circuit_manager import CircuitModeller


# SCRIPT_DIR = 'synbio_morpher', 'scripts'


# def get_all_config_files(config_dir):
#     exclude = ['params']
#     flat_configs = []
#     dirs = get_subdirectories(config_dir)
#     for config in get_pathnames(config_dir, allow_empty=True):
#         if config in dirs and config not in exclude:
#             flat_configs = flat_configs + get_all_config_files(config)
#         elif '.json' in config:
#             flat_configs.append(config)
#     return flat_configs


# class TestScripts(unittest.TestCase):

#     def test_script_input(self):
#         for script_home in get_subdirectories(SCRIPT_DIR):
#             for script_name in get_pathnames(file_key='run', search_dir=script_home, allow_empty=True):
#                 script_module = __import__(
#                     convert_pathname_to_module(script_name), fromlist=[''])
#                 script = getattr(script_module, 'main')
#                 if script.__name__ == 'wrapper':
#                     logging.warning(
#                         f'Could not test `main` function in {script_name} because it is wrapped.')
#                 else:
#                     self.assertIn(
#                         'config', inspect.getfullargspec(script).args,
#                         msg=f'Could not find "config" in script args "{script_name}"')
#                     self.assertIn(
#                         'data_writer', inspect.getfullargspec(script).args,
#                         msg=f'Could not find "data_writer" in script args "{script_name}"')

#     def test_script_baseconfig(self):
#         """ Test that every purpose in scripts that has a set of configs contain all the keys of the script"""
#         exclude = ['common', '__pycache__']

#         # base_config = {}
#         for script_home in get_subdirectories(SCRIPT_DIR):
#             if os.path.basename(script_home) in exclude:
#                 continue
#             # self.assertIn('configs', get_subdirectories(
#             #     script_home, only_basedir=True))
#             base_config = get_pathnames(file_key='base_config', search_dir=os.path.join(
#                 script_home, 'configs'), first_only=True, allow_empty=True)
#             base_config = load_json_as_dict(base_config) if base_config else {}
#             for config_path in get_all_config_files(os.path.join(script_home, 'configs')):
#                 config = load_json_as_dict(config_path)
#                 self.assertTrue(all(key in config.keys()
#                                 for key in base_config.keys()), msg=f'Keys in base_config {base_config.keys()} '
#                                 f'could not be found in config {config_path} {config.keys()}')
#                 # self.assertTrue(all(key in base_config.keys()
#                 #                 for key in config.keys()), msg=f'Keys in config {config_path} {config.keys()} '
#                 #                 f'could not be found in base_config {base_config.keys()}')
#                 # all(item in superset.items() for item in subset.items())

#     def test_circuit_configs(self):
#         """ Make sure that the keys required by the circuit composer are present in configs that have a data path. """
#         exclude = ['common', '__pycache__']

#         ESSENTIAL_KWARGS = ESSENTIAL_KWARGS + ["experiment"]

#         for script_home in get_subdirectories(SCRIPT_DIR):
#             if os.path.basename(script_home) in exclude:
#                 continue
#             for config_path in get_all_config_files(os.path.join(script_home, 'configs')):
#                 config = load_json_as_dict(config_path)
#                 if config.get("data") or config.get("data_path"):
#                     self.assertTrue(all(k in config.keys() for k in ESSENTIAL_KWARGS), msg=f'Config file {config_path} '
#                                     f'is missing some essential keys ({ESSENTIAL_KWARGS}) if a circuit is '
#                                     f'meant to be constructed with this config: {dict({k: k in config.keys() for k in ESSENTIAL_KWARGS})}')

#             base_config = get_pathnames(file_key='base_config', search_dir=os.path.join(
#                 script_home, 'configs'), first_only=True, allow_empty=True)
#             base_config = prepare_config(load_json_as_dict(base_config) if base_config else {})

#     def test_parameter_based_simulation(self):
#         config = {
#             "data_path": make_filename_safely("./synbio_morpher/scripts/parameter_based_simulation/configs/empty_circuit.fasta"),
#             "experiment": {
#                 "purpose": "parameter_based_simulation"
#             },
#             "interaction_simulator": {
#                 "name": "IntaRNA",
#                 "postprocess": True
#             },
#             "molecular_params": {
#                 "creation_rates": 50,
#                 "copynumbers": 5,
#                 "degradation_rates": 20
#             },
#             "system_type": "RNA"
#         }
#         data_writer = ResultWriter(purpose=config.get(
#             'experiment').get('purpose', 'parameter_based_simulation'))

#         interaction_array = np.arange(0, 1, 0.05)
#         size_interaction_array = np.size(interaction_array)

#         sample_names = {'RNA1': None, 'RNA2': None, 'RNA3': None}
#         num_species = len(sample_names)
#         num_unique_interactions = triangular_sequence(num_species)

#         matrix_dimensions = tuple(
#             [num_species] + [size_interaction_array]*num_unique_interactions)
#         matrix_size = num_species * \
#             np.power(size_interaction_array, num_unique_interactions)
#         self.assertEquals(matrix_size, np.prod(list(matrix_dimensions)),
#                           msg='Something is off about the intended size of the matrix')

#         all_species_steady_states = np.zeros(matrix_dimensions)

#         # For loop test
#         i = 11 + 4*size_interaction_array
#         flat_triangle = np.zeros(num_unique_interactions)
#         iterators = [int(np.mod(i / np.power(size_interaction_array, j),
#                                 size_interaction_array)) for j in range(num_unique_interactions)]
#         flat_triangle[:] = interaction_array[list(iterators)]
#         interaction_matrix = make_symmetrical_matrix_from_sequence(
#             flat_triangle, num_species)
#         cfg = {"interactions": {"interactions_matrix": interaction_matrix}}

#         circuit = construct_circuit_from_cfg(
#             extra_configs=cfg, config_file=config)
#         circuit = CircuitModeller(
#             result_writer=data_writer).find_steady_states(circuit)
#         idx = [slice(0, num_species)] + [[ite] for ite in iterators]
#         # all_species_steady_states[tuple(
#         #     idx)] = circuit.species.steady_state_copynums[:]
#         self.assertEquals(interaction_array[11], 0.55)
#         self.assertEquals(interaction_array[4], 0.2)
#         self.assertEquals(interaction_array[0], 0)
#         cfg = {"interactions": {
#             "interactions_matrix": np.array([
#                 [0.55, 0.2, 0],
#                 [0.2, 0, 0],
#                 [0, 0, 0]])}}

#         circuit = construct_circuit_from_cfg(
#             extra_configs=cfg, config_filepath=config)
#         circuit = CircuitModeller(
#             result_writer=data_writer).find_steady_states(circuit)
#         # self.assertEquals(all_species_steady_states[:, 11, 4, 0, 0, 0, 0][0],
#         #                   circuit.species.steady_state_copynums[0])
#         # self.assertEquals(all_species_steady_states[:, 11, 4, 0, 0, 0, 0][1],
#         #                   circuit.species.steady_state_copynums[1])
#         # self.assertEquals(all_species_steady_states[:, 11, 4, 0, 0, 0, 0][2],
#         #                   circuit.species.steady_state_copynums[2])        


# if __name__ == '__main__':
#     unittest.main()
