
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    

import os
from fire import Fire
from synbio_morpher.scripts.generate_species_templates.run_generate_species_templates import main as generate_species_templates
from synbio_morpher.scripts.gather_interaction_stats.run_gather_interaction_stats import main as gather_interaction_stats
from synbio_morpher.scripts.mutation_effect_on_interactions_signal.run_mutation_effect_on_interactions_signal import main as mutation_effect_on_interactions_signal
from synbio_morpher.scripts.summarise_simulation.run_summarise_simulation import main as summarise_simulation
from synbio_morpher.srv.io.manage.script_manager import script_preamble, ensemble_func


def main(config=None, data_writer=None):

    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        # "synbio_morpher", "scripts", "ensemble_mutation_effect_analysis", "configs", "test_ensemble.json"))
        "synbio_morpher", "scripts", "ensemble_mutation_effect_analysis", "configs", "test_large_scale.json"))
        # "synbio_morpher", "scripts", "ensemble_mutation_effect_analysis", "configs", "test_large_scale_loaded.json"))
        # "synbio_morpher", "scripts", "ensemble_mutation_effect_analysis", "configs", "test_small_loaded.json"))
        # "synbio_morpher", "scripts", "ensemble_mutation_effect_analysis", "configs", "distribution_of_energies.json"))
        # "synbio_morpher", "scripts", "ensemble_mutation_effect_analysis", "configs", "distribution_of_energies_loaded.json"))
        # "synbio_morpher", "scripts", "ensemble_mutation_effect_analysis", "configs", "test_small.json"))
        # "synbio_morpher", "scripts", "ensemble_mutation_effect_analysis", "configs", "test_small_vis.json"))

    ensemble_func(config, data_writer)


if __name__ == "__main__":
    Fire(main)
