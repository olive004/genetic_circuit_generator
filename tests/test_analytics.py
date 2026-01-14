import unittest
import numpy as np
import jax
from functools import partial
from copy import deepcopy


from synbio_morpher.utils.common.testing.shared import create_test_inputs, CONFIG, TEST_CONFIG
from synbio_morpher.utils.results.analytics.naming import get_true_names_analytics, RATIO_KEY
from synbio_morpher.utils.results.analytics.timeseries import generate_analytics
from synbio_morpher.utils.signal.signals_new import SignalFuncs


class TestAnalytics(unittest.TestCase):

    def test_interactions(self):

        num_species = 9
        fake_species = [str(i) for i in np.arange(num_species)]
        signal_idx = 0
        signal_onehot = 1 * \
            np.array([f == fake_species[signal_idx] for f in fake_species])

        t0 = 0
        t1 = 100
        t = np.arange(t0, t1)
        target = 100
        baseline = 20
        overshoot = 10
        overshoot_height = target + 10

        jax.config.update('jax_platform_name', 'cpu')

        data = np.asarray([
            SignalFuncs.step_function_integrated(t, t1/2 + i, target=target + i) +
            SignalFuncs.step_function(
                t, t1/2 + i, impulse_halfwidth=overshoot,
                target=overshoot_height) for i in range(num_species)]) + baseline
        ref_circuit_data = np.asarray([
            SignalFuncs.step_function_integrated(t, t1/2 + i * 2, target=target + i * 2) +
            SignalFuncs.step_function(
                t, t1/2 + i * 2, impulse_halfwidth=overshoot,
                target=overshoot_height) for i in range(num_species)]) + baseline
        data_rev = np.asarray([
            - SignalFuncs.step_function_integrated(t, t1/2 + i, target=baseline + i) -
            SignalFuncs.step_function(
                t, t1/2 + i, impulse_halfwidth=overshoot,
                target=overshoot_height) for i in range(num_species)]) + target
        ref_circuit_data_rev = np.asarray([
            - SignalFuncs.step_function_integrated(t, t1/2 + i * 2, target=baseline + i * 2) -
            SignalFuncs.step_function(
                t, t1/2 + i * 2, impulse_halfwidth=overshoot,
                target=overshoot_height) for i in range(num_species)]) + target

        analytics = {k: np.array(v) for k, v in
                     partial(generate_analytics, time=t, labels=fake_species,
                             signal_onehot=signal_onehot,
                             signal_time=t1/2,
                             ref_circuit_data=ref_circuit_data)(data).items()
                     }
        analytics_rev = {k: np.array(v) for k, v in
                         partial(generate_analytics, time=t, labels=fake_species,
                                 signal_onehot=signal_onehot,
                                 signal_time=t1/2,
                                 ref_circuit_data=ref_circuit_data_rev)(data_rev).items()
                         }

        # Signal in analytics
        self.assertEqual(
            analytics[f'precision'][0], 1)
        self.assertEqual(
            analytics[f'sensitivity'][0], 1)
        self.assertEqual(
            analytics_rev[f'precision'][0], 1)
        self.assertEqual(
            analytics_rev[f'sensitivity'][0], 1)
        for k in analytics.keys():
            if RATIO_KEY in k and 'deriv' not in k:
                self.assertTrue((analytics[k][0].astype(np.float16)[0] == 1) | (
                    analytics[k][0].astype(np.float16)[0] == np.inf))

        # Choice analytics
        [self.assertEqual(analytics['fold_change'].astype(np.float16)[
                          i], (target + baseline + i)/baseline) for i in range(num_species)]

        for n, c in zip(['c'], [CONFIG]):
            # for n, c in zip(['t', 'c'], [TEST_CONFIG, CONFIG]):
            circuits, config, data_writer, info = create_test_inputs(
                deepcopy(c))
            circuits[0].reset_to_initial_state()

            analytics_cols = get_true_names_analytics(
                candidate_cols=info.columns)

            if n == 'c':
                # Assert that steady states are correct
                info_ref = info[info['mutation_name'] == 'ref_circuit']
                # true_steady_states = [
                #     0.693848,
                #     0.667480,
                #     0.667480,
                #     0.586914,
                #     0.562988,
                #     1.486328,
                #     0.680176,
                #     0.649902,
                #     0.789062,
                #     0.858887,
                #     0.729004,
                #     0.771484,
                #     6.515625,
                #     2.634766,
                #     1.719727
                # ]
                # for i in range(len(true_steady_states)):
                #     self.assertEqual(info_ref['steady_states'].astype(
                #         np.float16).iloc[i], np.float16(true_steady_states[i]))

                # Test that 'weak' circuit had no interactions
                weak = info[info['circuit_name'] == '0_weak']
                weak = weak[weak['mutation_name'] == 'ref_circuit']
                # self.assertEquals(weak['fold_change'].unique(), 0)

            for s in info['sample_name'].unique():
                curr = info[info['sample_name'] == s]

                # Test that signal ratio is always 1 or inf given a ref circuit
                if s in c['signal']['inputs']:
                    curr_ref = curr[curr['circuit_name'] == 'ref_circuit']

                    for k in [a for a in analytics_cols if RATIO_KEY in a]:
                        self.assertTrue(all(curr_ref[k].astype(np.float16) == 1) | all(
                            curr_ref[k].astype(np.float16) == np.inf))

                # Test that if the max_amount ratio was 1, that steady states did not change

            # Test that columns that should be positive are positive
            positive_cols = [
                'fold_change',
                'overshoot',
                'max_amount',
                'min_amount',
                'RMSE',
                'steady_states',
                'response_time',
                'precision',
                'sensitivity'
            ]
            self.assertTrue(all(info[positive_cols] > 0))


def main():

    t = TestAnalytics()
    t.test_interactions()


if __name__ == '__main__':
    unittest.main()
