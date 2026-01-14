
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


from functools import partial

from synbio_morpher.utils.data.common import Data
from synbio_morpher.utils.misc.helper import none_func


class DataLoader():

    def __init__(self) -> None:
        pass

    def get_loader(self, filepath: str):
        from synbio_morpher.utils.data.data_format_tools.common import determine_file_format
        data_type = determine_file_format(filepath)
        loader = self.get_loader_by_dtype(data_type)
        return loader

    @staticmethod
    def get_loader_by_dtype(data_type):
        if data_type == 'fasta':
            from synbio_morpher.utils.data.data_format_tools.manipulate_fasta \
                import load_seq_from_FASTA
            return partial(load_seq_from_FASTA, as_type='dict')
        elif data_type == 'csv':
            from synbio_morpher.srv.io.loaders.misc import load_csv
            return load_csv
        elif data_type == 'numpy':
            import numpy as np
            return np.load
        elif data_type == None:
            return none_func
        else:
            raise NotImplementedError(
                "Other filetypes than fasta not supported yet.")

    def load_data(self, filepath: str):
        loader = self.get_loader(filepath)
        return loader(filepath)


class GeneCircuitLoader(DataLoader):

    def __init__(self) -> None:
        super().__init__()

    def load_data(self, filepath: str, identities: dict = {}) -> Data:
        return Data(super().load_data(filepath=filepath), source_files=filepath, identities=identities)
