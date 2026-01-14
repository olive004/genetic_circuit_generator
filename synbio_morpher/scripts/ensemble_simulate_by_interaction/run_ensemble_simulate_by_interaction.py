
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    

import os
from synbio_morpher.srv.io.manage.script_manager import script_preamble, ensemble_func


def main(config=None, data_writer=None):

    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "synbio_morpher", "scripts", "ensemble_simulate_by_interaction", "configs", "base_config.json"))

    ensemble_func(config, data_writer)
