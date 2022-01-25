from pathlib import Path

import pytest
import numpy as np

from icu_benchmarks.common.constants import STEPS_PER_HOUR, PAO2_MIX_SCALE, LEVEL1_RATIO_RESP, \
    LEVEL2_RATIO_RESP, LEVEL3_RATIO_RESP, VENT_ETCO2_TSH, FRACTION_TSH_RESP
from icu_benchmarks.endpoints import endpoint_benchmark

TEST_ROOT = Path(__file__).parent.parent

PREPROCESSING_RES = TEST_ROOT.parent / 'preprocessing' / 'resources'

MINS_PER_STEP = 60 // STEPS_PER_HOUR


def test_kernel_smooth_arr():

    # If fewer than 2 observations the unsmoothed array is returned as an edge case
    input_arr_1 = np.array([2])
    input_arr_1_bef=np.copy(input_arr_1)
    
    bandwidth_1 = 2
    res_1 = endpoint_benchmark.kernel_smooth_arr(input_arr_1, bandwidth_1)
    assert np.all(res_1 == input_arr_1)

    # Input array not corrupted
    assert np.all(input_arr_1_bef==input_arr_1)

    # Nadaraya Watson estimator is correctly applied

    input_arr_2 = np.array([1.3, 2.5, 3.6])
    bandwidth_2 = 2.5
    correct_formula = np.array([1.4436806438153837, 2.48934930210808, 3.468236127284816])

    res_2 = endpoint_benchmark.kernel_smooth_arr(input_arr_2, bandwidth_2)
    assert np.allclose(res_2, correct_formula)


@pytest.mark.parametrize("pf,vent,peep_status,peep_threshold,event",
                         ((LEVEL1_RATIO_RESP, 1.0, 1.0, 1.0, b"event_1"),
                          (LEVEL1_RATIO_RESP, 0.0, 1.0, 1.0, b"event_1"),
                          (LEVEL1_RATIO_RESP, 1.0, 0.0, 1.0, b"event_1"),
                          (LEVEL1_RATIO_RESP, 1.0, 1.0, 0.0, b"event_0"),
                          (LEVEL2_RATIO_RESP, 1.0, 1.0, 1.0, b"event_2"),
                          (LEVEL2_RATIO_RESP, 0.0, 1.0, 1.0, b"event_2"),
                          (LEVEL2_RATIO_RESP, 1.0, 0.0, 1.0, b"event_2"),
                          (LEVEL2_RATIO_RESP, 1.0, 1.0, 0.0, b"event_0"),
                          (LEVEL3_RATIO_RESP, 1.0, 1.0, 1.0, b"event_3"),
                          (LEVEL3_RATIO_RESP, 0.0, 1.0, 1.0, b"event_3"),
                          (LEVEL3_RATIO_RESP, 1.0, 0.0, 1.0, b"event_3"),
                          (LEVEL3_RATIO_RESP, 1.0, 1.0, 0.0, b"event_0"),
                          ))
def test_assign_resp_levels_fixed(pf, vent, peep_status, peep_threshold, event):
    # We test individually each level. Edges are tested in other tests
    pf_array = np.ones(64) * pf
    vent_array = np.ones(64) * vent
    peep_status_array = np.ones(64) * peep_status
    peep_threshold_array = np.ones(64) * peep_threshold
    off_set_window = 4
    event_window = 12
    out_array = endpoint_benchmark.assign_resp_levels(pf_array, vent_array,
                                                      event_window, peep_status_array,
                                                      peep_threshold_array, off_set_window)

    assert np.all(out_array[:-4] == event)
    assert np.all(out_array[-4:] == b'UNKNOWN')

    pf_array[:event_window] = np.nan
    out_array = endpoint_benchmark.assign_resp_levels(pf_array, vent_array,
                                                      event_window, peep_status_array,
                                                      peep_threshold_array, off_set_window)
    nan_bound = (int((1 - FRACTION_TSH_RESP) * event_window + 1))
    tsh_bound = int(FRACTION_TSH_RESP * event_window)

    assert np.all(out_array[:nan_bound] == b'UNKNOWN')
    if tsh_bound > nan_bound:
        assert np.all(out_array[nan_bound: tsh_bound] == b'event_0')


def test_percentile_smooth():
    # We test a flat signal
    flat_signal = np.ones((100,)).astype(float)
    out_flat = endpoint_benchmark.percentile_smooth(flat_signal, 50, 20)
    assert np.all(out_flat == flat_signal)

    # We test a constant slope signal
    steps_signal = np.arange(8).repeat(2).astype(float)
    out_steps = endpoint_benchmark.percentile_smooth(steps_signal, 50, 20)
    assert np.all(out_steps[:2] == 0.0)
    assert np.all(out_steps[2::2] - np.arange(7) == 0.5)
    assert out_steps[-1] == 7.0


def test_merge_short_vent_gaps():
    # TO DO: Positions which are not gaps are never modified - smth wrong with this
    # gaps of larger length than short_gap_hours are not removed

    vent_status_arr_1 = np.array(
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    short_gap_hours = 1

    status_1 = endpoint_benchmark.merge_short_vent_gaps(vent_status_arr_1, short_gap_hours)

    assert np.all(status_1 == vent_status_arr_1)

    # a gap of shorter length is removed
    vent_status_arr_2 = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    correct_status = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])

    status_2 = endpoint_benchmark.merge_short_vent_gaps(vent_status_arr_2, short_gap_hours)

    assert np.all(status_2 == correct_status)

    # The input array is not modified
    vent_status_arr_3 = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    status_3 = endpoint_benchmark.merge_short_vent_gaps(vent_status_arr_3, short_gap_hours)
    assert np.all(status_3 == vent_status_arr_3)

    # positions which are not gaps are never modified (zeros in the begginning os the array are not considered as gaps)
    vent_status_arr_4 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    status_4 = endpoint_benchmark.merge_short_vent_gaps(vent_status_arr_4, short_gap_hours)
    assert np.all(status_4 == vent_status_arr_4)


def test_delete_short_vent_events():
    # TO DO: verify about equal (!)

    # Short events shorted than the threshold are indeed deleted
    vent_status_arr_1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    short_event_hours_ = 1
    corrected_events = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    res_1 = endpoint_benchmark.delete_short_vent_events(vent_status_arr_1, short_event_hours_)
    assert np.all(res_1 == corrected_events)

    # Events longer or equal than the threshold are not deleted
    vent_status_arr_2 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    res_2 = endpoint_benchmark.delete_short_vent_events(vent_status_arr_2, short_event_hours_)

    assert np.all(res_2 == vent_status_arr_2)

    # Non-events are never modified
    vent_status_arr_3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    res_3 = endpoint_benchmark.delete_short_vent_events(vent_status_arr_3, short_event_hours_)
    assert np.all(res_3 == vent_status_arr_3)


def test_mix_real_est_pao2():
    # TO DO: The correct closest measurement is found in the first inner loop

    pao2_col_1 = np.array([200, 150, 170.1, 200.3])
    pao2_meas_cnt_1 = np.array([1, 4, 5, 6])
    pao2_est_arr_1 = np.array([198.3, 160.2, 175.3, 199.5])
    correct_formula = np.array([150.37022660280886, 150.07818449997205, 170.13985876469164, 200.29386788235516])

    pao2_col_2 = np.array([198.3, 198.3, 198.3, 198.3])
    pao2_meas_cnt_2 = np.array([1, 2, 3, 4])
    pao2_est_arr_2 = np.array([198.3, 198.3, 198.3, 198.3])

    # Save the inputs to check they are not modified by the function
    pao2_col_1_bef=np.copy(pao2_col_1)
    pao2_meas_cnt_1_bef=np.copy(pao2_meas_cnt_1)
    pao2_est_arr_1_bef=np.copy(pao2_est_arr_1)

    # formula is correct
    status_1 = endpoint_benchmark.mix_real_est_pao2(pao2_col_1, pao2_meas_cnt_1, pao2_est_arr_1)
    assert np.all(status_1 == correct_formula)

    # Input arrays not corrupted
    assert np.all(pao2_col_1_bef==pao2_col_1)
    assert np.all(pao2_meas_cnt_1_bef==pao2_meas_cnt_1)
    assert np.all(pao2_est_arr_1_bef==pao2_est_arr_1)

    # array doesn't change
    status_2 = endpoint_benchmark.mix_real_est_pao2(pao2_col_2, pao2_meas_cnt_2, pao2_est_arr_2)
    assert np.all(status_2 == pao2_est_arr_2)


def test_correct_right_edge_l0():
    # Event blocks are never modified if they are not adjacent to a level 0 block.
    # TO DO: verify

    offset_back_windows_ = 1

    # The right edge of an event0 block is corrected if required by the pf_event_est_arr values
    event_status_arr_1 = [b"event_0", b"event_0", b"event_0", b"event_0", b"event_1", b"event_1", b"event_1"]
    pf_event_est_arr_1 = np.array([350, 343, 362, 310, 306, 288, 263])
    correction = [b'event_0', b'event_0', b'event_0', b'event_0', b'event_0', b'event_1', b'event_1']

    status_1 = endpoint_benchmark.correct_right_edge_l0(event_status_arr_1, pf_event_est_arr_1, offset_back_windows_)
    assert np.all(status_1 == correction)

    # the right edge of an event0 block is not modified if the values in pf_event_est_arr do not indicate this
    event_status_arr_2 = [b"event_0", b"event_0", b"event_0", b"event_0", b"event_1", b"event_1", b"event_1"]
    pf_event_est_arr_2 = np.array([250, 243, 262, 210, 206, 288, 263])
    status_2 = endpoint_benchmark.correct_right_edge_l0(event_status_arr_2, pf_event_est_arr_2, offset_back_windows_)
    assert np.all(status_2 == event_status_arr_2)


def test_correct_right_edge_l1():
    # Event blocks are never modified if they are not adjacent to a level 0 block.
    # TO DO: verify

    offset_back_windows_ = 1

    # The right edge of an event1 block is corrected if required by the pf_event_est_arr values
    event_status_arr_1 = [b"event_1", b"event_1", b"event_1", b"event_1", b"event_2", b"event_2", b"event_2"]
    pf_event_est_arr_1 = np.array([350, 343, 362, 310, 299, 288, 263])
    correction = [b'event_1', b'event_1', b'event_1', b'event_1', b'event_1', b'event_1', b'event_2']

    status_1 = endpoint_benchmark.correct_right_edge_l1(event_status_arr_1, pf_event_est_arr_1, offset_back_windows_)
    assert np.all(status_1 == correction)

    # The right edge of an event1 block is not modified if the values in pf_event_est_arr do not indicate this
    event_status_arr_3 = [b"event_1", b"event_1", b"event_1", b"event_1", b"event_2", b"event_2", b"event_2"]
    pf_event_est_arr_3 = np.array([263, 273, 222, 188, 177, 163, 155])
    status_3 = endpoint_benchmark.correct_right_edge_l1(event_status_arr_3, pf_event_est_arr_3, offset_back_windows_)
    assert np.all(status_3 == event_status_arr_3)


def test_correct_right_edge_l2():
    offset_back_windows_ = 1

    # The right edge of an event1 block is corrected if required by the pf_event_est_arr values
    event_status_arr_1 = [b"event_2", b"event_2", b"event_2", b"event_2", b"event_1", b"event_1", b"event_1"]
    pf_event_est_arr_1 = np.array([188, 169, 155, 210, 190, 230, 225])
    correction = [b'event_2', b'event_2', b'event_2', b'event_2', b'event_2', b'event_1', b'event_1']

    status_1 = endpoint_benchmark.correct_right_edge_l2(event_status_arr_1, pf_event_est_arr_1, offset_back_windows_)
    assert np.all(status_1 == correction)

    # The right edge of an event1 block is not modified if the values in pf_event_est_arr do not indicate this
    event_status_arr_2 = [b"event_2", b"event_2", b"event_2", b"event_2", b"event_1", b"event_1", b"event_1"]
    pf_event_est_arr_2 = np.array([188, 169, 155, 210, 220, 230, 225])
    status_2 = endpoint_benchmark.correct_right_edge_l2(event_status_arr_2, pf_event_est_arr_2, offset_back_windows_)
    assert np.all(status_2 == event_status_arr_2)


def test_correct_right_edge_l3():
    offset_back_windows_ = 1

    # The right edge of an event1 block is corrected if required by the pf_event_est_arr values
    event_status_arr_1 = [b"event_3", b"event_3", b"event_3", b"event_2", b"event_2", b"event_2", b"event_1",
                          b"event_1"]
    pf_event_est_arr_1 = np.array([89, 99, 95, 92, 110, 160, 210, 220])
    correction = [b'event_3', b'event_3', b'event_3', b'event_3', b'event_2', b'event_2', b'event_1', b'event_1']
    status_1 = endpoint_benchmark.correct_right_edge_l3(event_status_arr_1, pf_event_est_arr_1, offset_back_windows_)
    assert np.all(status_1 == correction)

    # The right edge of an event1 block is not modified if the values in pf_event_est_arr do not indicate this
    event_status_arr_2 = [b"event_3", b"event_3", b"event_3", b"event_2", b"event_2", b"event_2", b"event_1",
                          b"event_1"]
    pf_event_est_arr_2 = np.array([89, 99, 95, 192, 110, 160, 210, 220])
    status_2 = endpoint_benchmark.correct_right_edge_l3(event_status_arr_2, pf_event_est_arr_2, offset_back_windows_)
    assert np.all(status_2 == event_status_arr_2)


def test_delete_small_continuous_blocks():
    # zero length block
    event_arr_0 = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1])
    block_threshold_zero = 0
    result_0 = endpoint_benchmark.delete_small_continuous_blocks(event_arr_0, block_threshold_zero)[0]
    assert np.all(event_arr_0 == result_0)

    # very big block
    event_arr_big = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1])
    block_threshold_big = 1000000000000000
    result_big = endpoint_benchmark.delete_small_continuous_blocks(event_arr_big, block_threshold_big)[0]
    assert np.all(event_arr_big == result_big)

    # a sandwiched block between two events of the same label is correctly changed,
    # if it has smaller or equal length than the threshold

    event_arr_1 = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1])
    block_threshold = 3
    result_1 = endpoint_benchmark.delete_small_continuous_blocks(event_arr_1, block_threshold)[0]

    assert np.all(result_1 == np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

    # A sandwiched block between two events of the same label is not changed, if its lengt is longer than the threshold

    event_arr_2 = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1])
    result_2 = endpoint_benchmark.delete_small_continuous_blocks(event_arr_2, block_threshold)[0]

    assert np.all(event_arr_2 == result_2)

    # blocks which are adjacent to two blocks of different labels are never modified
    event_arr_3 = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2])
    result_3 = endpoint_benchmark.delete_small_continuous_blocks(event_arr_3, block_threshold)[0]
    assert np.all(event_arr_3 == result_3)

    # a block is never modified if its neighbouring block are not both longer than it
    event_arr_4 = np.array([1, 1, 1, 2, 2, 2, 1, 1])
    result_4 = endpoint_benchmark.delete_small_continuous_blocks(event_arr_4, block_threshold)[0]
    assert np.all(event_arr_4 == result_4)

    # a block is never modified if both of its neighbouring blocks is not longer than the block length threshold
    event_arr_5 = np.array([1, 1, 1, 2, 2, 2, 1, 1])
    result_5 = endpoint_benchmark.delete_small_continuous_blocks(event_arr_5, block_threshold)[0]
    assert np.all(event_arr_5 == result_5)
