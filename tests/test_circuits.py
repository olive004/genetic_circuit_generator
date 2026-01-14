import unittest
import numpy as np
import logging


from synbio_morpher.utils.common.testing.shared import create_test_inputs, CONFIG, TEST_CONFIG
from synbio_morpher.srv.sequence_exploration.sequence_analysis import load_tabulated_info
from synbio_morpher.utils.results.analytics.naming import get_true_names_analytics, DIFF_KEY, RATIO_KEY
from synbio_morpher.utils.misc.type_handling import flatten_listlike


def fake_binding_site(energy):
    """ Let's say -100 kcal corresponds to the entire sequence binding """
    SEQ_LENGTH = 20
    if energy == 0:
        return [], 0, 0
    num_bs = np.min([SEQ_LENGTH, int(100 / np.abs(energy))])
    num_groups = np.random.randint(int(SEQ_LENGTH/num_bs)) + 1

    starting_bind = [np.max([1, np.random.randint(
        0, np.max([1, SEQ_LENGTH - num_bs - num_groups]))])]
    [starting_bind.append(starting_bind[-1] + np.random.randint(int(num_bs / num_groups),
                          SEQ_LENGTH - starting_bind[-1] - int(num_bs / num_groups))) for i in range(num_groups - 1)]

    fwd_bind = [np.arange(sb, sb+np.floor(num_bs/len(starting_bind)), 1)
                for sb in starting_bind]
    rev_bind = [SEQ_LENGTH - fb for fb in fwd_bind]
    return flatten_listlike([[(f, r) for f, r in zip(fwd, rev)] for fwd, rev in zip(fwd_bind, rev_bind)]), num_groups, num_bs


class TestCircuit(unittest.TestCase):

    def test_interactions(self):
        pass

    def test_refcircuits(self):
        """ Make sure that each simulation used the correct reference circuit """

        for c in [CONFIG]: # [TEST_CONFIG, CONFIG]:
            circuits, config, data_writer, info = create_test_inputs(c)

            ref_circuits = info[info["mutation_name"] == "ref_circuit"]

            self.assertEqual(len(ref_circuits.groupby('circuit_name').agg(
                str)), len(ref_circuits['circuit_name'].unique()))

            self.assertEqual(ref_circuits["mutation_num"].unique()[0], 0,
                             f'The reference circuits should have no mutations.')
            self.assertEqual(ref_circuits["RMSE"].unique()[0], 0,
                             f'The reference circuits should have an RMSE of 0.')
            self.assertEqual(ref_circuits["mutation_type"].unique()[0], '[]',
                             f'The reference circuits should have no mutations.')

            analytics_cols = get_true_names_analytics(
                candidate_cols=info.columns)
            diffs = info.set_index(['circuit_name', 'sample_name', 'mutation_name']).groupby('mutation_name', group_keys=False)[analytics_cols].apply(
                lambda x: x - ref_circuits.set_index(['circuit_name', 'sample_name'])[analytics_cols]).reset_index()
            ratios = info.set_index(['circuit_name', 'sample_name', 'mutation_name']).groupby('mutation_name', group_keys=False)[analytics_cols].apply(
                lambda x: x / ref_circuits.set_index(['circuit_name', 'sample_name'])[analytics_cols]).reset_index()

            # Make sure the reference circuits have no difference to their original values
            self.assertTrue(
                all([(diff == 0) or (np.isnan(diff)) for diff in sorted(
                    diffs[diffs['mutation_name'] == 'ref_circuit'][analytics_cols].agg('unique'))]),
                'The reference circuit should have a difference of 0 or nan to itself.')

            self.assertTrue(
                all([(ratio == 1) or (np.isnan(ratio)) for ratio in sorted(
                    ratios[ratios['mutation_name'] == 'ref_circuit'][analytics_cols].agg('unique'))]),
                'The reference circuit should have a ratio of 1 or nan to itself.')

            for col in analytics_cols:
                if DIFF_KEY in col:
                    self.assertEqual(ref_circuits[col].unique()[0], 0,
                                     f'The reference circuits should not be different to the base circuit since they are the base circuit.')
                elif RATIO_KEY in col:
                    self.assertTrue(np.sum((1 - ref_circuits[col].unique()) > 1e4) == 0 or (ref_circuits[col].unique()[0] == np.inf),
                                    f'The reference circuits should have a ratio of one or infinity to the base circuit since they are the base circuit.')
                elif col + DIFF_KEY in info.columns:
                    self.assertTrue(all(diffs[col] == info[col + DIFF_KEY]),
                                    'The difference between the reference circuit and its subcircuits should match between the computed and the tabulated numbers.')
                elif col + RATIO_KEY in info.columns:
                    self.assertTrue(all(ratios[col] == info[col + RATIO_KEY]),
                                    'The ratio between the reference circuit and its subcircuits should match between the computed and the tabulated numbers.')
                else:
                    logging.warning(
                        f'Column {col} should be in the table or exist there as {col + DIFF_KEY} or {col + RATIO_KEY} - columns are {info.columns}')

    def test_numerical_results(self):

        for c in [TEST_CONFIG, CONFIG]:
            circuits, config, data_writer, info = create_test_inputs(c)

            info

    def test_tabulation(self):

        for c, d in zip(
            [TEST_CONFIG, CONFIG],
            [{'debug_mode': 1},
             {'normal_mode': 1}]
        ):
            circuits, config, data_writer, info = create_test_inputs(c)

            source_dirs = [data_writer.ensemble_write_dir]
            info2 = load_tabulated_info(source_dirs)

            if info.isnull().any().all():
                logging.warning(
                    f'There are NaNs in the circuit summary table. Rethink.')

            self.assertTrue((info == info2).all().all(),
                            f'Loaded info table incorrectly - there are probably unwanted NaNs')  # This won't work for nan's
            self.assertTrue(all(info == info2),
                            f'Loaded info table incorrectly')


def main():

    t = TestCircuit()
    t.test_refcircuits()


if __name__ == '__main__':
    # unittest.main()
    main()
