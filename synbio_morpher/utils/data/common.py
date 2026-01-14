
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Optional, List
import pandas as pd


class Data():
    """ Holds things like FASTA files or other genetic info files. """

    def __init__(self, loaded_data: Optional[dict], identities: dict = {}, source_files: Optional[str] = None) -> None:
        self.data = loaded_data if loaded_data is not None else {}
        self.source = source_files
        self.sample_names = self.make_sample_names()
        self.identities = self.convert_names_to_idxs(
            identities, self.sample_names)

    def get_data_by_idx(self, idx):
        if idx not in self.sample_names:
            return self.data[self.sample_names[idx]]
        else:
            return self.data[idx]

    @staticmethod
    def convert_names_to_idxs(names_table: dict, source: list) -> dict:
        """ Output and input species definition """
        indexed_identities = {}
        for name_type, name in names_table.items():
            if name in source:
                indexed_identities[name_type] = source.index(name)
        if not indexed_identities and names_table and source:
            logging.warning(
                f'Identities not found: {names_table.values()} not in {source}')
        return indexed_identities

    def make_sample_names(self, sample_names: Optional[List[str]] = None) -> List[str]:
        if type(self.data) == dict:
            return list(self.data.keys())
        elif type(self.data) == pd.DataFrame:
            return list(self.data.columns)
        elif self.data is None and (sample_names is not None):
            return sample_names
        raise ValueError(f'Unrecognised loaded data type {type(self.data)}.')

    @property
    def size(self):
        if len(self.data) == len(self.sample_names):
            return len(self.data)
        else:
            logging.warning('Number of samples could be inaccurate - sample names '
                            f'{self.sample_names} different to number of data points ({len(self.data)})')
            return max(len(self.data), len(self.sample_names))

    @size.getter
    def size(self):
        if len(self.data) == len(self.sample_names):
            return len(self.data)
        else:
            logging.warning('Number of samples could be inaccurate - sample names '
                            f'{self.sample_names} different to number of data points ({len(self.data)})')
            return max(len(self.data), len(self.sample_names))

    @property
    def sample_names(self):
        return self._sample_names

    @sample_names.getter
    def sample_names(self) -> List[str]:
        return self.make_sample_names()
        # if type(self.data) == dict:
        #     return list(self.data.keys())
        # else:
        #     import numpy as np
        #     return list(range(len(self.data)))

    @sample_names.setter
    def sample_names(self, value):
        if type(value) == list or type(value) == dict:
            self._sample_names = value
        else:
            raise ValueError(f'Wrong type "{type(value)}" for ' +
                             'setting data sample_names to {value}')
