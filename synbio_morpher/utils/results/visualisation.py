
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Union, Optional, Sequence
import networkx as nx
import numpy as np
import pandas as pd
import logging
from synbio_morpher.utils.misc.type_handling import merge_dicts
from synbio_morpher.utils.results.writer import DataWriter
from synbio_morpher.utils.misc.string_handling import add_outtype, make_time_str, prettify_keys_for_label
from synbio_morpher.utils.misc.database_handling import select_rows_by_conditional_cols
import seaborn as sns
# import matplotlib
# matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
plt.ioff()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def hist3d(data, x, y, log_scale=[False, False, False]):
    import plotly.graph_objects as go

    if (type(x) == str) and (type(y) == str):
        xv = data[x]
        yv = data[y]
    else:
        xv, yv = x, y
    h = np.histogram2d(xv, yv, bins=50) # type: ignore

    def bin_centers(bins):
        centers = (bins + (bins[1]-bins[0])/2)[:-1]
        return centers

    x_bins_centers = bin_centers(h[1])
    y_bins_centers = bin_centers(h[2])

    if log_scale[2]:
        d = np.log(h[0])
    else:
        d = h[0]
    df = pd.DataFrame(d, index=x_bins_centers, columns=y_bins_centers)
    fig = go.Figure(data=[go.Surface(z=df)])

    if log_scale[0]:
        fig.update_xaxes(type="log")
    if log_scale[1]:
        fig.update_yaxes(type="log")
    return fig


def visualise_data(data: pd.DataFrame, data_writer: DataWriter,
                   cols_x: Optional[Sequence] = None, 
                   cols_y: Optional[Sequence] = None,
                   groupby: Optional[str] = None,
                   normalise_data_x: bool = False,
                   normalise_data_y: bool = False,
                   plot_type: Optional[str] = None,
                   out_name='test_plot', out_type='svg',
                   hue: Optional[str] = None,
                   preprocessor_func_x=None,
                   postprocessor_func_x=None,
                   postprocessor_func_y=None,
                   use_sns=False,
                   log_axis=[False, False],
                   threshold_value_max=None,
                   bin_count=100,
                   selection_conditions: Optional[List[tuple]] = None,
                   remove_outliers_y: bool = False,
                   outlier_std_threshold_y=3,
                   exclude_rows_nonempty_in_cols: Optional[list] = None,
                   exclude_rows_zero_in_cols: Optional[list] = None,
                   plot_cols_on_same_graph=False,
                   misc_histplot_kwargs=None,
                   **plot_kwargs):
    """ Plot type can be any attributes of VisODE() 
    This function is a mess - only for visualisation.
    Args:
    cols_x and cols_y: list of columns. If a column is a tuple, the other columns in the tuple
    will be treated as increasingly nested subcolumns """

    if groupby is not None and type(groupby) == str:
        import operator
        selection_conditions = [
            (groupby, operator.eq, data[groupby].unique()[0])]

    def process_cols(cols):
        if cols is None:
            return [None]
        return cols

    cols_x = process_cols(cols_x)
    cols_y = process_cols(cols_y)
    
    cols_x = cols_x if cols_x is not None else [None]

    def process_log_sns(data: pd.DataFrame, column_x: Union[tuple, str], column_y: Union[tuple, str],
                        plot_kwrgs: dict, log_axis: list, plot_type: Optional[str] = None):
        exempt_from_log = ['histplot']

        def process_log(data, col, log_option, col_label):
            if log_option:
                data = data[data[col] != 0]
                data = data.drop(col, axis=1).join(data[col].abs())
                log_l = r'$Log_{10}$' + ' of '
                if type(col_label) == tuple:
                    col_label = list(col_label)
                    new_col = tuple([log_l + col_label[0]] + col_label[1:])
                else:
                    new_col = r'$Log_{10}$' + ' of ' + col_label
                if plot_type not in exempt_from_log:
                    log_df = np.log10(data[col])
                    data = data.drop(col, axis=1).join(log_df)
            else:
                new_col = col_label

            if type(col) == tuple:
                assert type(
                    new_col) == tuple, 'If using multi-indexing, make sure new column names are also input as tuples'
                data = data.rename(
                    columns={c: n for c, n in zip(col, new_col)})
            else:
                data = data.rename(columns={col: new_col})
            return data, new_col
 
        if column_x is not None:
            data, column_x = process_log(
                data, column_x, log_axis[0], plot_kwrgs.get('xlabel', column_x))
        if column_y is not None:
            data, column_y = process_log(
                data, column_y, log_axis[1], plot_kwrgs.get('ylabel', column_y))
        if column_x is None and column_y is None:
            logging.warning('No columns given as input to histplot.')
        data.reset_index(drop=True, inplace=True)
        return data, column_x, column_y

    def preprocess_data(data: pd.DataFrame, preprocessor_func_x,
                        threshold_value_max,
                        column_x: Union[tuple, str], 
                        column_y: str,
                        normalise_data_x: bool, normalise_data_y: bool,
                        exclude_rows_nonempty_in_cols, exclude_rows_zero_in_cols,
                        preprocessor_func_y=None,
                        postprocessor_func_x=None,
                        postprocessor_func_y=None,
                        selection_conditions=None):
        def preprocess(data, column, normalise_data,
                       preprocessor_func, remove_outliers=False,
                       outlier_std_threshold=3):
            if column is not None:
                if preprocessor_func:
                    data = data.drop(column, axis=1).join(
                        preprocessor_func(data[column]))
                if threshold_value_max:
                    data = data[data[column] <= threshold_value_max]
                if remove_outliers:
                    mean = data[column].mean()
                    std = data[column].std()
                    if std != 0:
                        data = data[data[column] < mean +
                                    np.multiply(outlier_std_threshold, std)]
                        data = data[data[column] > mean -
                                    np.multiply(outlier_std_threshold, std)]
                if normalise_data:
                    df = data[column]
                    s = (df - df.mean()) / df.std()
                    data = data.drop(column, axis=1).join(s)

            return data, column

        def postprocess(data: pd.DataFrame, column: Union[str, tuple], postprocessor_func):
            if postprocessor_func:
                data = data.drop(column, axis=1).join(
                    postprocessor_func(data[column]))
            return data

        data, column_x = preprocess(
            data, column_x, normalise_data_x,
            preprocessor_func_x)
        data, column_y = preprocess(
            data, column_y, normalise_data_y,
            preprocessor_func_y, remove_outliers_y, outlier_std_threshold_y)
        data.reset_index(drop=True, inplace=True)

        for exc_cols, condition in zip(
            [exclude_rows_nonempty_in_cols, exclude_rows_zero_in_cols],
                ['', 0]):
            data = exclude_rows_by_cols(data, exc_cols, condition)

        if selection_conditions is not None:
            data = select_rows_by_conditional_cols(data, selection_conditions)
        data = postprocess(data, column_x, postprocessor_func_x)
        data = postprocess(data, column_y, postprocessor_func_y)
        data.reset_index(drop=True, inplace=True)
        return data, column_x, column_y

    def exclude_rows_by_cols(data: pd.DataFrame, cols: list, condition):
        if cols is not None:
            for exc_col in cols:
                data = data[data[exc_col] != condition]
        data.reset_index(drop=True, inplace=True)
        return data

    def is_data_plotable(data: pd.DataFrame, col_x: Union[tuple, str], col_y: Union[tuple, str]):
        if data.empty:
            return False
        if col_x is not None and ((data[col_x].all() and data[col_x].iloc[0] is np.nan) or data[col_x].isnull().all()):
            return False
        if col_y is not None and ((data[col_y].all() and data[col_y].iloc[0] is np.nan) or data[col_y].isnull().all()):
            return False
        return True

    visualiser = VisODE()
    if plot_type == 'plot':
        for col_x in cols_x:
            for col_y in cols_y:
                x, y = data[col_x], data[col_y]
                if preprocessor_func_x:
                    x = preprocessor_func_x(x.values)
                    # y = preprocessor_func_x(y.values)
                data_writer.output(out_type='png', out_name=out_name,
                                   write_func=visualiser.plot,
                                   **merge_dicts({'data': x, 'y': y}, plot_kwargs))
        return

    for col_x in cols_x:
        for col_y in cols_y:
            if use_sns:
                data, col_x, col_y = preprocess_data(
                    data, preprocessor_func_x, threshold_value_max,
                    col_x, col_y,
                    normalise_data_x, normalise_data_y,
                    exclude_rows_nonempty_in_cols, exclude_rows_zero_in_cols,
                    selection_conditions=selection_conditions,
                    postprocessor_func_x=postprocessor_func_x,
                    postprocessor_func_y=postprocessor_func_y)
                data, col_x, col_y = process_log_sns(
                    data, col_x, col_y, plot_kwargs, log_axis, plot_type)
                if plot_type == 'scatter_plot':
                    write_func = visualiser.sns_scatterplot
                elif plot_type == 'bar_plot':
                    write_func = visualiser.sns_barplot
                elif plot_type == 'violin_plot':
                    write_func = visualiser.sns_violinplot
                elif plot_type == 'line_plot':
                    write_func = visualiser.sns_lineplot
                elif plot_type == 'histplot':
                    write_func = visualiser.sns_histplot
                    plot_kwargs.update({'log_axis': log_axis,
                                        'bin_count': bin_count,
                                        'histplot_kwargs': misc_histplot_kwargs
                                        })
                else:
                    raise ValueError(f'Unknown plot type given "{plot_type}"')

                if is_data_plotable(data, col_x, col_y):
                    data_writer.output(out_type=out_type, out_name=out_name,
                                       write_func=write_func,
                                       **merge_dicts({'x': col_x, 'y': col_y, 'data': data,
                                                      'hue': hue},
                                                     plot_kwargs))
                else:
                    logging.warning(
                        f'Could not visualise columns {col_x} and {col_y}')


def visualise_graph_pyvis(graph: nx.DiGraph,
                          out_path: str,
                          new_vis=False,
                          out_type='html'):
    import webbrowser
    import os
    from synbio_morpher.utils.results.graph import NetworkCustom

    if new_vis:
        out_path = f'{out_path}_{make_time_str()}'
    out_path = add_outtype(out_path, out_type)

    interactive_graph = NetworkCustom(
        height='800px', width='800px', directed=True, notebook=True)
    interactive_graph.from_nx(graph, edge_weight_transf=lambda x: round(x, 7))
    interactive_graph.inherit_edge_colors(True)
    interactive_graph.set_edge_smooth('dynamic')
    try:
        interactive_graph.save_graph(out_path)
    except PermissionError:
        logging.warning(
            f'PermissionError: The graph at {out_path} may not have saved properly.')

    # web_filename = 'file:///' + os.getcwd() + '/' + out_path
    # # webbrowser.open(web_filename, new=1, autoraise=True)


def visualise_graph_pyplot(graph: nx.DiGraph):
    ax1 = plt.subplot(111)
    nx.draw(graph)
    raise NotImplementedError


class VisODE():
    def __init__(self, figsize=(10, 8)) -> None:
        self.figsize = figsize

    def add_kwrgs(self, plt, **plot_kwrgs):
        def warning_call(object, function, dummy_value, **dummy_kwargs):
            logging.warn(
                f'Could not call function {function} in object {object}.')

        for plot_kwrg, value in plot_kwrgs.items():
            if value is not None:
                if type(value) == dict:
                    # Treat as kwargs
                    getattr(plt, plot_kwrg, partial(
                        warning_call, object=plt, dummy_value=None, function=plot_kwrg))(**value)
                else:
                    f = getattr(plt, plot_kwrg, partial(
                        warning_call, dummy_value=None, function=plot_kwrg))
                    if callable(f):
                        f(value)

    def rename_legend(self, plot, new_title: str, new_labels: list):
        leg = plot.axes.flat[0].get_legend()
        leg.set_title(new_title)
        for t, l in zip(leg.texts, new_labels):
            t.set_text(l)

    def plot(self, data, y=None, new_vis=False, t=None, out_path='test_plot', out_type='svg',
             subsample=True,
             **plot_kwrgs) -> None:

        data = data.T if len(plot_kwrgs.get('legend', [])
                             ) == np.shape(data)[0] else data

        if type(data) != np.ndarray or type(t) != np.ndarray:
            t = np.array(t)
            data = np.array(data)

        if subsample:
            target_vis_size = 5000  # Max time steps to plot
            if data.shape[0] > target_vis_size:
                step_sz = int(data.shape[0] / target_vis_size)
                if len(data.shape) > 1:
                    data = data[::step_sz, :]
                else:
                    data = data[::step_sz]
                t = t[::step_sz]
                if y is not None:
                    y = y[::step_sz]

        plt.figure()
        if y is not None:
            plt.plot(data, y)
        elif t is not None:
            # if np.shape(t)[0] != np.shape(data)[0] and len(np.shape(data)) == 2:
            #     logging.warning()
            plt.plot(t, data)
        else:
            plt.plot(data)
        self.add_kwrgs(plt, **plot_kwrgs)
        plt.savefig(out_path)
        plt.close()

    def sns_generic_plot(self, plot_func, out_path: str, 
                         x: Optional[str], y: Optional[str], data: pd.DataFrame,
                         title: Optional[str] = None,
                         figsize=(7, 5), style="ticks",
                         **plot_kwargs):
        sns.set_context('paper')
        sns.set_theme(style=style) # type: ignore
        sns.set_palette("viridis")
        if title is None:
            title = plot_kwargs.get('title')
        if plot_kwargs.get('hue') is not None:
            new_hue = prettify_keys_for_label(plot_kwargs['hue'])
            data = data.rename(
                columns={plot_kwargs.get('hue'): new_hue})
            plot_kwargs['hue'] = new_hue
            pal = plot_kwargs.get('palette', sns.color_palette("viridis", len(data[plot_kwargs.get('hue')].unique())))
            plot_kwargs.update({
                # 'palette': sns.color_palette("husl", len(data[plot_kwargs.get('hue')].unique()))
                # 'palette': sns.color_palette("plasma", len(data[plot_kwargs.get('hue')].unique()))
                'palette': pal
            })
        else:
            if 'palette' in plot_kwargs:
                plot_kwargs.pop('palette')

        if x is None:
            logging.warning(
                'Make sure a column is specified for visualising with seaborn')
        f, ax = plt.subplots(figsize=figsize)

        try:
            plot_func(
                data=data,
                x=x,
                y=y,
                **plot_kwargs
            ).set(title=title)
            if plot_kwargs.get('hue') and not (data[x].min() is np.nan and data[x].max() is np.nan):
                if ax.get_legend() is not None:
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            f.savefig(out_path, bbox_inches='tight')
            plt.close()
        except KeyError as error:

            if all(data[x].unique() == np.inf):
                logging.warning(
                    f'All values infinity - could not plot {x} and {y} for plot: {title}')
                plt.close()
            elif plot_kwargs.get('hue'):
                logging.warning(
                    f'Failed - attempting to plot {x} without {plot_kwargs.get("hue")}')
                plot_kwargs.pop('hue')
                self.sns_generic_plot(
                    plot_func, out_path, x, y, data, title, figsize, style, **plot_kwargs)
            else:
                logging.warning(error)

    def sns_barplot(self, out_path: str, x: str, y: str, data: pd.DataFrame,
                    title: Optional[str] = None, xlabel=None, ylabel=None,
                    **plot_kwargs):
        # plot_kwargs.update({'errwidth': 1})

        # if plot_kwargs.get('hue'):
        #     plot_kwargs.update({
        #         # 'palette': sns.color_palette("husl", len(data[plot_kwargs.get('hue')].unique()))
        #         # 'palette': sns.color_palette("plasma", len(data[plot_kwargs.get('hue')].unique()))
        #         'palette': sns.color_palette("viridis", len(data[plot_kwargs.get('hue')].unique()))
        #     })

        self.sns_generic_plot(sns.barplot, out_path, x,
                              y, data, title, **plot_kwargs)

    def sns_scatterplot(self, out_path: str, x: str, y: str, data: pd.DataFrame,
                        title: Optional[str] = None, xlabel=None, ylabel=None,
                        **plot_kwargs):
        plot_kwargs['palette'] = plot_kwargs.get('palette', 'viridis')
        self.sns_generic_plot(sns.scatterplot, out_path,
                              x, y, data, title, **plot_kwargs)

    def sns_violinplot(self, out_path: str, x: str, y: str, data: pd.DataFrame,
                       title: Optional[str] = None, xlabel=None, ylabel=None,
                       **plot_kwargs):
        self.sns_generic_plot(sns.violinplot, out_path, x, y, data, title,
                              figsize=(10, 6), **plot_kwargs)

    def sns_lineplot(self, out_path: str, x: str, y: str, data: pd.DataFrame,
                     title: Optional[str] = None, xlabel=None, ylabel=None,
                     **plot_kwargs):
        if plot_kwargs.get('hue'):
            plot_kwargs.update({
                # 'palette': sns.color_palette("husl", len(data[plot_kwargs.get('hue')].unique()))
                # 'palette': sns.color_palette("plasma", len(data[plot_kwargs.get('hue')].unique()))
                'palette': sns.color_palette("viridis", len(data[plot_kwargs.get('hue')].unique()))
            })

        self.sns_generic_plot(sns.lineplot, out_path,
                              x, y, data, title, **plot_kwargs)

    # Make visualisations for each analytic chosen
    def heatmap(self, data: pd.DataFrame, out_path: str, out_type='png',
                new_vis=False, vmin=None, vmax=None, **plot_kwrgs):
        sns.set_theme()

        fig = plt.figure(figsize=self.figsize)
        ax = sns.heatmap(data, vmin=vmin, vmax=vmax)
        self.add_kwrgs(plt, **plot_kwrgs)
        fig.savefig(out_path)
        plt.clf()

    def sns_histplot(self, data: pd.DataFrame, out_path,
                     bin_count=100,
                     x: Optional[str] = None, y: Optional[str] = None,
                     log_axis: tuple = (False, False),
                     histplot_kwargs: Optional[dict] = None,
                     **plot_kwargs):
        """ log_axis: use logarithmic axes in (x-axis, y-axis) """
        sns.set_context('paper')

        histplot_kwargs = {} if histplot_kwargs is None else histplot_kwargs
        default_kwargs = {
            "bins": bin_count,
            "edgecolor": ".3",
            # element='poly'
            # "element": "step",
            "log_scale": log_axis,
            "fill": True,
            "linewidth": .5,
            "multiple": "dodge",  # "stack", "fill", "dodge"
            "palette": "deep",
            # palette="light:m_r",
            "title": plot_kwargs.get('title'),
            "hue": plot_kwargs.get('hue')
        }
        default_kwargs.update(histplot_kwargs)
        self.sns_generic_plot(sns.histplot, out_path,
                              x, y, data, **default_kwargs)

    def histplot(self, data: pd.DataFrame, out_path,
                 bin_count=100,
                 **plot_kwrgs):
        logging.warning(f'Function "histplot" is not finished.')
        plt.figure()
        plt.hist(data, range=(min(data.min()), max(data.max())), bins=bin_count)
        self.add_kwrgs(plt, **plot_kwrgs)
        plt.savefig(out_path)
        plt.close()
