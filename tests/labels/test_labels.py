from pathlib import Path

import pytest
import numpy as np

from icu_benchmarks.labels import label_benchmark_util

STEPS_PER_HOUR = label_benchmark_util.STEPS_PER_HOUR
TEST_ROOT = Path(__file__).parent.parent

PREPROCESSING_RES = TEST_ROOT.parent / 'preprocessing' / 'resources'


@pytest.mark.parametrize("lhours,rhours", ((0, 4),
                                           (0, 8),
                                           (2, 4),
                                           (2, 8),
                                           (0, 12),
                                           ))
def test_transition_to_failure(lhours, rhours):
    annotated_col = np.concatenate([np.zeros(2 * rhours * STEPS_PER_HOUR),
                                    np.ones(rhours * STEPS_PER_HOUR),
                                    np.zeros(2 * rhours * STEPS_PER_HOUR),
                                    np.ones(lhours * STEPS_PER_HOUR),
                                    np.zeros(2 * lhours * STEPS_PER_HOUR)])
    start_r_failure, end_r_failure, = 2 * rhours * STEPS_PER_HOUR, 3 * rhours * STEPS_PER_HOUR
    start_l_failure, end_l_failure, = 5 * rhours * STEPS_PER_HOUR, 5 * rhours * STEPS_PER_HOUR + lhours * STEPS_PER_HOUR

    labels = label_benchmark_util.transition_to_failure(annotated_col, lhours, rhours)

    # Check we no label during failure
    if start_r_failure != end_r_failure:
        assert np.all(np.isnan(labels[start_r_failure:end_r_failure]))
    if start_l_failure != end_l_failure:
        assert np.all(np.isnan(labels[start_l_failure:end_l_failure]))

    # Check correct labeling around first long event
    if start_r_failure != end_r_failure:
        assert np.all(labels[start_r_failure - rhours * STEPS_PER_HOUR:start_r_failure] == 1)
        assert np.all(labels[0:start_r_failure - rhours * STEPS_PER_HOUR] == 0)
        assert np.all(labels[end_r_failure:end_r_failure + rhours * STEPS_PER_HOUR] == 0)

    # Check correct labeling around second short event
    if start_l_failure != end_l_failure:
        positive_l = labels[start_l_failure - rhours * STEPS_PER_HOUR:start_l_failure - lhours * STEPS_PER_HOUR]
        if len(positive_l) > 0:
            assert np.all(positive_l == 1)
        start_l_neg = start_l_failure - end_l_failure - lhours * STEPS_PER_HOUR - 1
        assert np.all(labels[min(start_l_failure, start_l_neg):start_l_failure] == 0)


@pytest.mark.parametrize("target,lhours,rhours", ((0.5, 0, 4),
                                                  (0.5, 0, 8),
                                                  (0.5, 2, 4),
                                                  (0.5, 2, 8),
                                                  (0.5, 0, 12)
                                                  ))
def test_transition_to_abs(target, lhours, rhours):
    annotated_col = np.concatenate([np.zeros(2 * rhours * STEPS_PER_HOUR),
                                    np.ones(rhours * STEPS_PER_HOUR),
                                    np.zeros(2 * rhours * STEPS_PER_HOUR),
                                    np.ones(lhours * STEPS_PER_HOUR),
                                    np.zeros(2 * lhours * STEPS_PER_HOUR)])
    start_r_failure, end_r_failure, = 2 * rhours * STEPS_PER_HOUR, 3 * rhours * STEPS_PER_HOUR
    start_l_failure, end_l_failure, = 5 * rhours * STEPS_PER_HOUR, 5 * rhours * STEPS_PER_HOUR + lhours * STEPS_PER_HOUR

    labels = label_benchmark_util.transition_to_abs(annotated_col, target, lhours, rhours)

    # Check we no label during failure
    if start_r_failure != end_r_failure:
        assert np.all(np.isnan(labels[start_r_failure:end_r_failure]))
    if start_l_failure != end_l_failure:
        assert np.all(np.isnan(labels[start_l_failure:end_l_failure]))

    # Check correct labeling around first long event
    if start_r_failure != end_r_failure:
        assert np.all(labels[start_r_failure - rhours * STEPS_PER_HOUR:start_r_failure] == 1)
        assert np.all(labels[0:start_r_failure - rhours * STEPS_PER_HOUR] == 0)
        assert np.all(labels[end_r_failure:end_r_failure + rhours * STEPS_PER_HOUR] == 0)

    # Check correct labeling around second short event
    if start_l_failure != end_l_failure:
        positive_l = labels[start_l_failure - rhours * STEPS_PER_HOUR:start_l_failure - lhours * STEPS_PER_HOUR]
        if len(positive_l) > 0:
            assert np.all(positive_l == 1)
        start_l_neg = start_l_failure - end_l_failure - lhours * STEPS_PER_HOUR - 1
        assert np.all(labels[min(start_l_failure, start_l_neg):start_l_failure] == 0)


@pytest.mark.parametrize("stay_length_hours,mort_status,at_hours", ((24, 1, 24),
                                                                    (24, 0, 24),
                                                                    (25, 1, 24),
                                                                    (12, 0, 24),
                                                                    (12, 1, 24)))
def test_dynamic_mortality_at_hours(stay_length_hours, mort_status, at_hours):
    stay_length = stay_length_hours * STEPS_PER_HOUR
    labels = label_benchmark_util.dynamic_mortality_at_hours(stay_length, mort_status, at_hours)
    assert len(labels) == stay_length
    if stay_length >= at_hours * STEPS_PER_HOUR:  # Stay longer than at_hours should have a unique label
        assert labels[at_hours * STEPS_PER_HOUR - 1] == mort_status
        assert len(labels[~np.isnan(labels)]) == 1
    else:  # Others shouldn't
        assert np.all(np.isnan(labels))


@pytest.mark.parametrize("rhours", ((2,)))
def test_future_urine_output(rhours):
    stay_length = 12 * rhours * STEPS_PER_HOUR
    urine_col = np.zeros(stay_length)
    urine_meas_arr = np.zeros(stay_length)
    weight = np.ones(stay_length) * 60.0
    weight[2 * rhours * STEPS_PER_HOUR + 1:] = 30.0  # We devide weight by 2 after first measurement
    idxs_meas = [2 * rhours * STEPS_PER_HOUR, 6 * rhours * STEPS_PER_HOUR]
    for idx in idxs_meas:
        urine_meas_arr[idx] = 1
        urine_col[idx - rhours * STEPS_PER_HOUR + 1:idx + 1] = 60.0  # We set rate to 1 and 2 ml/h/kg
        assert len(urine_col[idx - rhours * STEPS_PER_HOUR + 1:idx + 1]) == rhours * STEPS_PER_HOUR
    labels_reg, labels_binary = label_benchmark_util.future_urine_output(urine_col, urine_meas_arr, weight, rhours)

    # Check labels are only rhours before measurements
    assert np.all(np.where(~np.isnan(labels_reg)) == np.array(idxs_meas) - rhours * STEPS_PER_HOUR)
    assert np.all(np.where(~np.isnan(labels_binary)) == np.array(idxs_meas) - rhours * STEPS_PER_HOUR)

    # Check the labels are positive for binary
    assert np.all(labels_binary[~np.isnan(labels_binary)] == 1.0)
    assert np.all(labels_reg[~np.isnan(labels_reg)] == np.array([1.0, 2.0]))
