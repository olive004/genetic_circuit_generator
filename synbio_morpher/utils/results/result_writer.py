
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


from typing import Dict, Union
import numpy as np

from synbio_morpher.utils.results.analytics.naming import get_analytics_types_all
from synbio_morpher.utils.results.results import Result
from synbio_morpher.utils.results.writer import DataWriter
from synbio_morpher.utils.misc.numerical import transpose_arraylike
from synbio_morpher.utils.misc.string_handling import make_time_str
from synbio_morpher.utils.circuit.agnostic_circuits.circuit import Circuit


class ResultWriter(DataWriter):
    def __init__(self, purpose, out_location=None) -> None:
        super().__init__(purpose, out_location)

    def make_metric_visualisation(self, result, keys, source: dict, new_report: bool):
        for plottable in keys:
            if plottable in source.keys():
                out_name = f'{result.name}_{plottable}'
                result.vis_kwargs.update({
                    'data': source.get(plottable),
                    'new_vis': new_report
                })
                self.output(out_name=out_name,
                            write_func=result.vis_func, **result.vis_kwargs)

    @staticmethod
    def make_report(keys: list, source: dict):

        def prettify_writeable(writeable):
            if type(writeable) != str:
                # if type(writeable) == np.ndarray or (type(writeable) == list and type(writeable[0]) != str):
                writeable = np.array(writeable)
                if writeable.ndim == 2 and np.shape(writeable)[1] == 1:
                    writeable = np.squeeze(writeable)
                writeable = list(writeable)
            elif type(writeable) == dict:
                for k, v in writeable.items(): # type: ignore
                    writeable[k] = prettify_writeable(v)
            return writeable

        report = {}
        for writeable in keys:
            if writeable not in source:
                writeable = [s for s in source.keys() if writeable in s]
            if type(writeable) != list:
                writeable = [writeable]
            for w in writeable:
                report[w] = prettify_writeable(source.get(w, ''))

        return report

    def write_report(self, writeables: list, analytics: dict, new_report: bool, out_name: str = 'report', out_type: str = 'json'):
        report = self.make_report(writeables, analytics)

        if new_report:
            out_name = out_name + '_' + make_time_str()
        self.output(out_type=out_type, out_name=out_name, overwrite=not (
            new_report), data=report)

    def write_numerical(self, data, out_name: str):
        self.output(out_type='csv', out_name=out_name,
                    data=transpose_arraylike(data), overwrite=True)

    def write_analytics(self, result: Result, new_report=False):
        analytics = result.analytics
        if analytics is not None:
            writeables = get_analytics_types_all()
            self.write_report(writeables, analytics, new_report,
                            out_name=f'report_{result.name}')

    def write_results(self, results: Dict[str, Result], new_report=False, no_visualisations=False,
                      only_numerical=False, no_analytics=False, no_numerical=False):

        for _name, result in results.items():
            if result.no_write:
                continue

            if not no_visualisations and (result.vis_kwargs is not None):
                result.vis_kwargs.update(
                    {'new_vis': new_report, 'data': result.data})

            if not no_numerical:
                self.write_numerical(
                    data=result.data, out_name=result.name + '_data')
            if not only_numerical:
                if not no_visualisations:
                    self.visualise(out_name=result.name,
                                   writer=result.vis_func, vis_kwargs=result.vis_kwargs)
                    if result.analytics is not None:
                        plottables = ['first_derivative']
                        self.make_metric_visualisation(
                            result, plottables, result.analytics, new_report)
                if not no_analytics:
                    self.write_analytics(result, new_report=new_report)

    def write_all(self, circuit: Circuit, new_report: bool, no_visualisations: bool = False,
                  only_numerical: bool = False, no_numerical: bool = False, no_analytics: bool = False):
        if not no_visualisations:
            self.visualise_graph(circuit)
        self.write_results(circuit.result_collector.results,
                           new_report=new_report, no_visualisations=no_visualisations,
                           only_numerical=only_numerical, no_numerical=no_numerical,
                           no_analytics=no_analytics)

    def visualise(self, out_name, writer, vis_kwargs):
        self.output(out_name=out_name, write_func=writer, **vis_kwargs)

    def visualise_graph(self, circuit: Circuit, mode="pyvis", new_vis=False):
        """ This is broken right now - networkx has updated """
        return
        # from synbio_morpher.utils.results.graph import Graph
        # # input_species = sorted(set(flatten_listlike(
        # #     [r.input for r in circuit.model.reactions if r.output and r.input])))
        # # idxs = [circuit.model.species.index(i) for i in input_species]
        # # input_eqconstants = np.asarray([
        # #     [circuit.interactions.eqconstants[i, ii] for ii in idxs] for i in idxs])
        # input_eqconstants = circuit.interactions.eqconstants
        # input_species = sorted(set(flatten_listlike([
        #     r.input for r in circuit.model.reactions if r.input])))
        # graph = Graph(source_matrix=input_eqconstants,
        #               labels=sorted([s.name for s in input_species]))

        # out_path = os.path.join(self.write_dir, 'graph')
        # if mode == 'pyvis':
        #     from synbio_morpher.utils.results.visualisation import visualise_graph_pyvis
        #     visualise_graph_pyvis(graph=graph.graph,
        #                           out_path=out_path, new_vis=new_vis)
        # else:
        #     from synbio_morpher.utils.results.visualisation import visualise_graph_pyplot
        #     visualise_graph_pyplot(graph=graph.graph, new_vis=new_vis)
