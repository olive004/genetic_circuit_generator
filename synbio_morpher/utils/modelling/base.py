
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


class Modeller():
    """ Common modeller class. For example deterministic vs. 
    stochastic modelling share some things """

    def __init__(self, max_time=0, time_interval=1, t0=0) -> None:
        """ Time step is the dt """
        self.t0 = t0
        self.max_time = int(max_time / time_interval)
        self.original_max_time = max_time
        self.time_interval = time_interval

    def dxdt_RNA(self, t, copynumbers, full_interactions, creation_rates, degradation_rates):
        pass
