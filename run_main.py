import logging
from fire import Fire
# from tests.test_circuits import main
# from tests.test_analytics import main
# from synbio_morpher.scripts.agnostic_simulation.run_agnostic_simulation import main
# from synbio_morpher.scripts.RNA_circuit_simulation.run_RNA_circuit import main
# from synbio_morpher.utils.data.fake_data_generation.nc_sequences import main
# from synbio_morpher.scripts.pair_species_mutation.run_pair_species_mutation import main
# from synbio_morpher.scripts.generate_species_templates.run_generate_species_templates import main
# from synbio_morpher.scripts.generate_seqs_flexible.run_generate_seqs_flexible import main
# from synbio_morpher.scripts.ensemble_generate_circuits.run_ensemble_generate_circuits import main
# from synbio_morpher.scripts.gather_interaction_stats.run_gather_interaction_stats import main
# from synbio_morpher.scripts.mutation_effect_on_interactions_signal.run_mutation_effect_on_interactions_signal import main
# from synbio_morpher.scripts.summarise_simulation.run_summarise_simulation import main
# from synbio_morpher.scripts.analyse_mutated_templates.run_analyse_mutated_templates import main
# from synbio_morpher.scripts.vis_0_histplots.run_vis_0_histplots import main
# from synbio_morpher.scripts.vis_0_histplots_nosig.run_vis_0_histplots_nosig import main
# from synbio_morpher.scripts.vis_1_histplots_interactions.run_vis_1_histplots_interactions import main
# from synbio_morpher.scripts.vis_2_histplots_diffs_ratios.run_vis_2_histplots_diffs_ratios import main
# from synbio_morpher.scripts.vis_2_histplots_diffs_ratios_nosig.run_vis_2_histplots_diffs_ratios_nosig import main
# from synbio_morpher.scripts.vis_3_mutation_types_positions_1.run_vis_3_mutation_types_positions_1 import main
# from synbio_morpher.scripts.vis_3_mutation_types_positions_2.run_vis_3_mutation_types_positions_2 import main
# from synbio_morpher.scripts.vis_4_histplots_thresholded.run_vis_4_histplots_thresholded import main
# from synbio_morpher.scripts.vis_5_means_analytics.run_vis_5_means_analytics import main
# from synbio_morpher.scripts.ensemble_mutation_effect_analysis.run_ensemble_mutation_effect_analysis import main
# from synbio_morpher.scripts.ensemble_visualisation.run_ensemble_visualisation import main
# from synbio_morpher.scripts.parameter_based_simulation.run_parameter_based_simulation import main
# from synbio_morpher.scripts.stitch_parameter_grid.run_stitch_parameter_grid import main
# from synbio_morpher.scripts.parameter_grid_analysis.run_parameter_grid_analysis import main
# from synbio_morpher.scripts.parameter_grid_analysis.run_multi_parameter_grid_analysis import main
# from synbio_morpher.scripts.mc_evolution.run_mc_evolution import main
from synbio_morpher.scripts.ensemble_simulate_by_interaction.run_ensemble_simulate_by_interaction import main


FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    Fire(main)
