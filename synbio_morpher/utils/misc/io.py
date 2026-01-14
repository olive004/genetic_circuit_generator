
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from typing import List, Union, Optional
import glob
import os
from synbio_morpher.utils.misc.string_handling import remove_file_extension
from synbio_morpher.utils.misc.type_handling import nest_list_dict
from synbio_morpher.utils.misc.helper import vanilla_return
from synbio_morpher.utils.misc.errors import ExperimentError
from synbio_morpher.utils.data.data_format_tools.common import load_multiple_as_list


def isolate_filename(filepath: Optional[str]):
    if type(filepath) == str:
        return os.path.splitext(os.path.basename(filepath))[0]
    return None


def create_location(pathname):
    if not os.path.isdir(pathname):
        os.umask(0)
        os.makedirs(pathname, mode=0o777)


def get_pathnames_from_mult_dirs(search_dirs: List[str], **get_pathnames_kwargs):
    return load_multiple_as_list(search_dirs, get_pathnames, **get_pathnames_kwargs)


def get_pathnames(search_dir: str, file_key: Union[List, str] = '', first_only: bool = False,
                  allow_empty: bool = False, subdir: str = '',
                  subdirs: Optional[list] = None,
                  conditional: Union[str, None] = 'filenames',
                  as_dict=False):
    """ Get the pathnames in a folder given a keyword. 

    Args:
    conditional: can be 'filenames' (default) to only return pathnames of files, 
    'directories' to only return pathnames of directories, or None to return all pathnames
    """
    path_condition_f = vanilla_return
    if conditional == 'directories':
        path_condition_f = os.path.isdir
    elif conditional == 'filenames':
        path_condition_f = os.path.isfile

    path_names = []
    if type(file_key) == list:
        if as_dict and subdirs is not None:
            all_path_names = {}
            for fk, sd in zip(file_key, subdirs):
                curr_search_dir = os.path.join(
                    search_dir, sd) if sd is not None else search_dir
                all_path_names[fk] = sorted(list(set([f for f in glob.glob(os.path.join(
                    curr_search_dir, '*' + fk + '*')) if path_condition_f(f)])))
            path_names = nest_list_dict(all_path_names)
        else:
            all_path_names = []
            for fk in file_key:
                all_path_names.append(
                    set([f for f in glob.glob(os.path.join(
                        search_dir, '*' + fk + '*')) if path_condition_f(f)])
                )
            # all_path_names = flatten_nested_listlike(all_path_names)
            path_names = list(
                all_path_names[0].intersection(*all_path_names[1:]))
    elif not file_key:
        path_names = sorted([os.path.join(search_dir, f) for f in os.listdir(
            search_dir) if path_condition_f(os.path.join(search_dir, f))])
    elif type(file_key) == str:
        path_names = sorted([f for f in glob.glob(os.path.join(
            search_dir, '*' + file_key + '*')) if path_condition_f(f)])
    if first_only and path_names:
        path_names = path_names[0]
    if not path_names and subdir:
        path_names = get_pathnames(
            search_dir=os.path.join(search_dir, subdir), file_key=file_key, as_dict=as_dict,
            first_only=first_only, allow_empty=allow_empty, subdir=subdir, subdirs=subdirs,
            conditional=conditional)
    if not path_names and not allow_empty:
        raise ExperimentError(
            f'Could not find file matching "{file_key}" in {search_dir}.')
    return path_names


def get_subdirectories(parent_dir, only_basedir=False, min_content_in_dir: int = 0) -> list:
    subdirectories = [f.path for f in os.scandir(
        parent_dir) if f.is_dir() and len(os.listdir(f.path)) > min_content_in_dir]
    if only_basedir:
        return sorted([os.path.basename(s) for s in subdirectories])
    return sorted(subdirectories)


def convert_pathname_to_module(filepath: str):
    filepath = remove_file_extension(filepath)
    return os.path.normpath(filepath).replace(os.sep, '.')


def import_module_from_path(module_name: str, filepath: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is not None:
        return importlib.util.module_from_spec(spec)
    else:
        raise ModuleNotFoundError(f'Could not find module {module_name} in {filepath}')
