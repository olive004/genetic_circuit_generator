
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


from copy import deepcopy
from typing import Dict, Optional, Any
import logging
import numpy as np
from synbio_morpher.utils.misc.type_handling import assert_uniform_type


class Result():
    def __init__(self, name: str, result_data: Optional[np.ndarray], category: str, vis_func,
                 time: Optional[np.ndarray] = None, vis_kwargs: Optional[dict] = None, 
                 analytics_kwargs: Optional[dict] = None,
                 analytics: Optional[Dict[str, Any]] = None, no_write: bool = False) -> None:
        self.name = name
        self.data = result_data
        self.category = category
        self.vis_func = vis_func
        self.vis_kwargs = vis_kwargs
        self.analytics_kwargs = analytics_kwargs
        self.analytics = analytics
        self.no_write = no_write
        if category == 'time_series' and analytics is None:
            from synbio_morpher.utils.results.analytics.timeseries import generate_analytics
            analytics_kwargs = analytics_kwargs if analytics_kwargs is not None else {}
            self.analytics = generate_analytics(
                result_data, time, **analytics_kwargs)
            #     result_data, time).generate_analytics(**analytics_kwargs)

    def __repr__(self):
        str_rep = [f'\n\nResult {self.name}\n']
        for k, v in self.__dict__.items():
            str_rep.append(f'{k}: {v}\n')
        return ''.join(str_rep)


class ResultCollector():

    def __init__(self) -> None:

        self.results: Dict[str, Result] = {}

    def add_result(self, data, category: str, vis_func, name: str,
                   time: Optional[np.ndarray] = None, vis_kwargs: Optional[dict] = None,
                   analytics_kwargs: Optional[dict] = None, analytics=None,
                   no_write: bool = False) -> None:
        """ category: 'time_series', 'graph' """
        name = f'Result_{len(self.results.keys())}' if not name else name
        result_entry = Result(name, data, category, vis_func, time,
                              vis_kwargs, analytics_kwargs, analytics,
                              no_write)
        self.results[name] = result_entry

    def delete_result(self, key, just_data=True):
        if key in self.results and just_data:
            self.results[key].data = None
        elif not just_data:
            del self.results[key]

    def delete_result_data(self):
        for key in self.results.keys():
            self.results[key].data = None

    def add_modified_duplicate_result(self, key, **add_kwargs):
        result = deepcopy(self.get_result(key))
        result_kwargs = result.__dict__
        result_kwargs.update(add_kwargs)
        self.add_result(**result_kwargs)

    def get_result(self, key) -> Result:
        result = self.results.get(key, None)
        if result is None:
            raise ValueError(
                f'The keyword {key} did not return a result from {self}.')
        return result

    def pool_results(self, results: dict):
        assert_uniform_type(results.values, Result)
        self.results.update(results)

    def reset(self):
        self.results = {}
