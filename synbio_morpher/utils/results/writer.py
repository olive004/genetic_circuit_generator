
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from typing import Optional
from copy import deepcopy
from functools import partial
from abc import ABC, abstractmethod
import os
import pandas as pd
import inspect
import importlib


from synbio_morpher.srv.io.manage.sys_interface import PACKAGE_NAME, SCRIPT_DIR, DATA_DIR
from synbio_morpher.utils.data.data_format_tools.common import write_csv, write_json, write_np
from synbio_morpher.utils.data.data_format_tools.manipulate_fasta import write_fasta_file
from synbio_morpher.utils.misc.io import create_location
from synbio_morpher.utils.misc.string_handling import add_outtype, make_time_str
from synbio_morpher.utils.misc.type_handling import find_sublist_max, convert_type_from_pandas


class DataWriter():

    def __init__(self, purpose: str, out_location: Optional[str] = None, ensemble_script: Optional[str] = None) -> None:
        self.purpose = purpose
        self.script_dir = SCRIPT_DIR
        self.root_output_dir = DATA_DIR
        self.exception_dirs = ['example', 'tests']

        # Top dir is the directory immediately below the script directory
        if out_location is None:
            self.top_write_dir = self.make_location_from_purpose(purpose)
        else:
            self.top_write_dir = out_location
        self.ensemble_script = ensemble_script
        self.ensemble_write_dir = deepcopy(self.top_write_dir) if self.ensemble_script is None \
            else os.path.join(self.top_write_dir, self.ensemble_script)
        create_location(self.ensemble_write_dir)
        self.write_dir = deepcopy(self.ensemble_write_dir)

    def output(self, out_name: str, out_type: Optional[str] = None, overwrite: bool = False, return_path: bool = False,
               new_file: bool = False, filename_addon: Optional[str] = None, subfolder: Optional[str] = None, write_master: bool = False,
               write_func=None, write_to_top_dir=False, **writer_kwargs):

        if out_type is None:
            raise ValueError(
                f'The out_type for file "{out_name}" needs to be specified for writing with function "{write_func}".')

        def make_base_name(filename_addon):
            if out_name is not None and (self.write_dir in out_name):
                base_name = os.path.basename(out_name)
            if new_file:
                if filename_addon is None:
                    filename_addon = make_time_str()
                base_name = f'{out_name}_{filename_addon}'
            else:
                base_name = f'{out_name}'
            return base_name

        def make_out_path(base_name) -> str:
            write_dir = self.write_dir if not write_to_top_dir else self.ensemble_write_dir
            if subfolder:
                out_subpath = os.path.join(write_dir, subfolder)
                create_location(out_subpath)
                out_path = os.path.join(
                    out_subpath, add_outtype(base_name, out_type))
            elif out_type is None:
                out_path = os.path.join(
                    write_dir, base_name)
            else:
                out_path = os.path.join(
                    write_dir, add_outtype(base_name, out_type))
            return out_path

        base_name = make_base_name(filename_addon)
        out_path = make_out_path(base_name)

        if write_func is None:
            write_func = self.get_write_func(
                out_type, out_path, overwrite=overwrite)
        writer_kwargs['out_path'] = out_path
        if write_master:
            if not os.path.exists(out_path):
                self.write_to_output_summary(
                    out_name, out_name=base_name, out_path=out_path,
                    filename_addon=filename_addon, out_type=out_type)

        write_func(**writer_kwargs)
        if return_path:
            return out_path

    def get_write_func(self, out_type: str, out_path: str, overwrite: bool):
        if out_type == "fasta":
            return partial(write_fasta_file, out_path=out_path)
        if out_type == "csv":
            return partial(write_csv, out_path=out_path, overwrite=overwrite)
        if out_type == "json":
            return partial(write_json, out_path=out_path, overwrite=overwrite)
        if out_type == "npy":
            return partial(write_np, out_path=out_path, overwrite=overwrite)
        raise ValueError(
            f'No write function available for output of type {out_type}')

    def make_location_from_purpose(self, purpose: str) -> str:
        try: 
            if purpose in importlib.import_module('.' +purpose, '.'.join([PACKAGE_NAME, SCRIPT_DIR])).__name__ or purpose in self.exception_dirs:
                location = os.path.join(self.root_output_dir,
                                        purpose,
                                        self.generate_location_instance())
                create_location(location)
                return location
            raise ValueError(
                f'Unrecognised purpose {purpose} for writing data to - make sure directory of this name exists in {".".join([PACKAGE_NAME, SCRIPT_DIR])}')
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f'Unrecognised purpose {purpose} for writing data to - make sure directory of this name exists in {".".join([PACKAGE_NAME, SCRIPT_DIR])}')

    def generate_location_instance(self):
        return make_time_str()

    def get_write_dir_aslist(self):
        return self.write_dir.split(os.sep)

    def subdivide_writing(self, name: str, safe_dir_change=True):
        """ Update the working write_dir to write to a new subdirectory. 
        If safe_dir_change is True, recursive subdividing can be avoided, 
        as the new subdirectory is added onto the top writer directory
        (ensemble_write_dir). """
        base_dir = self.ensemble_write_dir if safe_dir_change else self.write_dir
        location = os.path.join(base_dir, name)
        create_location(location)
        self.write_dir = location

    def update_ensemble(self, new_ensemble: str):
        old_ensemble_write_dir = deepcopy(self.ensemble_write_dir)
        self.ensemble_script = new_ensemble
        self.ensemble_write_dir = os.path.join(
            self.top_write_dir, self.ensemble_script)
        if not os.path.isdir(self.ensemble_write_dir):
            create_location(os.path.join(self.ensemble_write_dir))
        self._update_writedir(old_ensemble_write_dir)

    def _update_writedir(self, old_ensemble_write_dir):
        if old_ensemble_write_dir in self.write_dir:
            self.write_dir = self.write_dir.replace(
                old_ensemble_write_dir, self.ensemble_write_dir)
        elif self.top_write_dir in self.write_dir:
            self.write_dir = self.write_dir.replace(
                self.top_write_dir, self.ensemble_write_dir)
        else:
            raise ValueError(f'Cannot update write directory (currently {self.write_dir}) for DataWriter. '
                             f'Should contain either {old_ensemble_write_dir} or {self.top_write_dir}')

    def unsubdivide_last_dir(self):
        self.write_dir = os.path.dirname(self.write_dir)
        if self.ensemble_write_dir not in self.write_dir:
            self.unsubdivide()

    def reset_ensemble(self):
        old_ensemble_write_dir = deepcopy(self.ensemble_write_dir)
        self.ensemble_script = ''
        self.ensemble_write_dir = deepcopy(self.top_write_dir)
        self._update_writedir(old_ensemble_write_dir)

    def unsubdivide(self):
        self.write_dir = deepcopy(self.ensemble_write_dir)

    def write_to_output_summary(self, name: str, **kwargs):
        output_summary = {str(k): str(v) for k, v in kwargs.items()}
        output_summary['name'] = name
        output_summary['subdir'] = self.write_dir.replace(
            self.top_write_dir + os.sep, '')
        self.output(out_type='csv', out_name='output_summary', write_master=False,
                    **{'data': output_summary}) # type: ignore


class Tabulated(ABC):

    def __init__(self) -> None:
        self.column_names, self.data = self.get_table_properties()
        self.max_table_length = find_sublist_max(self.data)

    def as_table(self):
        return pd.DataFrame.from_dict(dict(
            zip(self.column_names, [[v] for v in self.data])),
            dtype=object)

    @abstractmethod
    def get_table_properties(self):
        props = self.__dict__
        return list(props.keys()), list(props.values())


def kwargs_from_table(tabulated_cls: Tabulated, table: pd.Series):
    """ Reverse of as_table. Table should only have one row """
    kwargs = {}
    for c in table.index:
        if c in inspect.getfullargspec(getattr(tabulated_cls, '__init__')).args:
            expected_type = inspect.getfullargspec(getattr(tabulated_cls, '__init__')).annotations[c]
            kwargs[c] = convert_type_from_pandas(table[c], expected_type)
    return kwargs
