
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


import logging
from typing import Optional
from synbio_morpher.srv.io.loaders.data_loader import GeneCircuitLoader
from synbio_morpher.utils.data.common import Data


class DataManager():
    def __init__(self, identities: dict, filepath: Optional[str] = None, data: Optional[dict] = None):
        self.loader = GeneCircuitLoader()
        self.source = filepath
        if data is None and filepath:
            self.data: Data = self.loader.load_data(filepath, identities=identities)
        elif data is None and filepath is None:
            self.data = Data(loaded_data=data, identities=identities)
            # logging.warning(
            #     f'No data: either a data filepath or raw data must be supplied to DataManager')
            # raise ValueError('Either a data filepath or raw data must be supplied.')
        else:
            self.data = Data(loaded_data=data, identities=identities)

    def __repr__(self) -> str:
        return str(self.data)
