
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Tuple, Any
import pandas as pd


def load_csv(filepath, load_as=None, return_header=False) -> Union[Any, Tuple[Any, list]]:

    og_csv_file = pd.read_csv(filepath)
    if load_as == 'numpy':
        csv_file = og_csv_file.to_numpy()
    elif load_as == 'dict':
        csv_file = og_csv_file.to_dict()
    else:
        csv_file = og_csv_file
    if return_header:
        return csv_file, list(og_csv_file.columns)
    return csv_file
