""" Functions to generate the set of endpoints for the time series
    benchmark on the HiRID database"""

import glob
import logging
import math
import os
import os.path
import pickle
import sys

import numpy as np
import pandas as pd
import skfda.preprocessing.smoothing.kernel_smoothers as skks
import skfda.representation.grid as skgrid
from icu_benchmarks.common.constants import STEPS_PER_HOUR, LEVEL1_RATIO_RESP, LEVEL2_RATIO_RESP, LEVEL3_RATIO_RESP, \
    FRACTION_TSH_CIRC, FRACTION_TSH_RESP, DATETIME, PID, REL_DATETIME, SPO2_NORMAL_VALUE, VENT_ETCO2_TSH, \
    NIV_VENT_MODE, SUPPOX_TO_FIO2


PAO2_MIX_SCALE = 57 ** 2
MINS_PER_STEP = 60 // STEPS_PER_HOUR
MAX_SUPPOX_KEY = np.array(list(SUPPOX_TO_FIO2.keys())).max()
MAX_SUPPOX_TO_FIO2_VAL = SUPPOX_TO_FIO2[MAX_SUPPOX_KEY]

def load_pickle(fpath):
    """ Given a file path pointing to a pickle file, yields the object pickled in this file"""
    with open(fpath, 'rb') as fp:
        return pickle.load(fp)

# UT : Mid
def mix_real_est_pao2(pao2_col, pao2_meas_cnt, pao2_est_arr):
    """ Mix real PaO2 measurement and PaO2 estimates using a Gaussian kernel
     
    INPUTS:
    pao2_col: Imputed PaO2 time series
    pao2_meas_cnt: Cumulative real measurement counts of PaO2
    pao2_est_arr: PaO2 estimates at each point computed by another function

    RETURNS: A 1D time series of the final mixed PaO2 estimate
    """
    final_pao2_arr = np.copy(pao2_est_arr)
    sq_scale = PAO2_MIX_SCALE  # 1 hour has mass 1/3 approximately

    for idx in range(final_pao2_arr.size):
        meas_ref = pao2_meas_cnt[idx]
        real_val = None
        real_val_dist = None

        # Search forward and backward with priority giving to backward if equidistant
        for sidx in range(4 * STEPS_PER_HOUR):
            if not idx - sidx < 0 and pao2_meas_cnt[idx - sidx] < meas_ref:
                real_val = pao2_col[idx - sidx + 1]
                real_val_dist = MINS_PER_STEP * sidx
                break
            elif not idx + sidx >= final_pao2_arr.size and pao2_meas_cnt[idx + sidx] > meas_ref:
                real_val = pao2_col[idx + sidx]
                real_val_dist = MINS_PER_STEP * sidx
                break

        if real_val is not None:
            alpha_mj = math.exp(-real_val_dist ** 2 / sq_scale)
            alpha_ej = 1 - alpha_mj
            final_pao2_arr[idx] = alpha_mj * real_val + alpha_ej * pao2_est_arr[idx]

    return final_pao2_arr

# UT : Mid
def kernel_smooth_arr(input_arr, bandwidth=None):
    """ Kernel smooth an input array with a Nadaraya-Watson kernel smoother
    
    INPUTS:
    input_arr: Input array to be smoothed

    RETURNS: Input array smoothed with kernel

    """
    output_arr = np.copy(input_arr)
    fin_arr = output_arr[np.isfinite(output_arr)]
    time_axis = MINS_PER_STEP * np.arange(len(output_arr))
    fin_time = time_axis[np.isfinite(output_arr)]

    # Return the unsmoothed array if fewer than 2 observations
    if fin_arr.size < 2:
        return output_arr

    smoother = skks.NadarayaWatsonSmoother(smoothing_parameter=bandwidth)
    fgrid = skgrid.FDataGrid([fin_arr], fin_time)
    fd_smoothed = smoother.fit_transform(fgrid)
    output_smoothed = fd_smoothed.data_matrix.flatten()
    output_arr[np.isfinite(output_arr)] = output_smoothed
    return output_arr

# UT : Mid
def percentile_smooth(signal_col, percentile, win_scope_mins):
    """ Window percentile smoother, where percentile is in the interval [0,100]
    
    INPUTS:
    signal_col: Input time series to be smoothed
    percentile: Which percentile should be used to take from the centralized window?
    win_scope_mins: Length of smoothing windows in minutes

    RETURNS: Smoothed input array
    """
    out_arr = np.zeros_like(signal_col)
    mins_per_window = MINS_PER_STEP
    search_range = int(win_scope_mins / mins_per_window / 2)
    for jdx in range(out_arr.size):
        search_arr = signal_col[max(0, jdx - search_range):min(out_arr.size, jdx + search_range)]
        out_arr[jdx] = np.percentile(search_arr, percentile)
    return out_arr

# UT : HIGH
# TODO: Refactor
def merge_short_vent_gaps(vent_status_arr, short_gap_hours):
    """ Merge short gaps in the ventilation status array
    
    INPUTS: 
    vent_status_arr: Binary ventilation status estmate at each time-step
    short_gap_hours: All gaps which are less than or equal to this threshold will be merged

    RETURNS: Ventilator status array with gaps removed

    """
    in_gap = False
    gap_length = 0

    for idx in range(len(vent_status_arr)):
        cur_state = vent_status_arr[idx]
        if in_gap and (cur_state == 0.0 or np.isnan(cur_state)):
            gap_length += MINS_PER_STEP
        elif not in_gap and (cur_state == 0.0 or np.isnan(cur_state)):
            in_gap = True
            in_gap_idx = idx
            gap_length = MINS_PER_STEP
        elif in_gap and cur_state == 1.0:
            in_gap = False
            if gap_length / 60. <= short_gap_hours:
                vent_status_arr[in_gap_idx:idx] = 1.0

    return vent_status_arr

# UT : VERY HIGH
def assign_resp_levels(event_status_arr=None, pf_event_est_arr=None, vent_status_arr=None,
                       ratio_arr=None, sz_window=None, peep_status_arr=None, peep_threshold_arr=None,
                       offset_back_windows=None):
    """Now label based on the array of estimated Horowitz indices

    INPUTS: 
    event_status_arr: The array to be filled with the ventilation status (OUTPUT)
    pf_event_est_arr: Array of estimated P/F ratios at each time step
    vent_status_arr: Array of estimated ventilation status at each time-step
    ratio_arr: Passed for length compute
    sz_window: Size of forward windows specified as multiple of time series gridding size
    peep_status_arr: Status array of available PEEP measurement at a time-point
    peep_threshold_arr: Binary array indicating if PEEP threshold was reached at a time-point
    offset_back_windows: Do not compute respiratory level for incomplete windows at the end of stay

    RETURNS: Filled event status array
    """
    for idx in range(0, len(event_status_arr) - offset_back_windows):
        est_idx = pf_event_est_arr[idx:min(len(ratio_arr), idx + sz_window)]
        est_vent = vent_status_arr[idx:min(len(ratio_arr), idx + sz_window)]
        est_peep_dense = peep_status_arr[idx:min(len(ratio_arr), idx + sz_window)]
        est_peep_threshold = peep_threshold_arr[idx:min(len(ratio_arr), idx + sz_window)]

        if np.sum((est_idx <= LEVEL3_RATIO_RESP) & (
                (est_vent == 0.0) | (est_vent == 1.0) & (est_peep_dense == 0.0) | (est_vent == 1.0) & (
                est_peep_dense == 1.0) & (est_peep_threshold == 1.0))) >= FRACTION_TSH_RESP * len(est_idx):
            event_status_arr[idx] = "event_3"
        elif np.sum((est_idx <= LEVEL2_RATIO_RESP) & (
                (est_vent == 0.0) | (est_vent == 1.0) & (est_peep_dense == 0.0) | (est_vent == 1.0) & (
                est_peep_dense == 1.0) & (est_peep_threshold == 1.0))) >= FRACTION_TSH_RESP * len(est_idx):
            event_status_arr[idx] = "event_2"
        elif np.sum((est_idx <= LEVEL1_RATIO_RESP) & (
                (est_vent == 0.0) | (est_vent == 1.0) & (est_peep_dense == 0.0) | (est_vent == 1.0) & (
                est_peep_dense == 1.0) & (est_peep_threshold == 1.0))) >= FRACTION_TSH_RESP * len(est_idx):
            event_status_arr[idx] = "event_1"
        elif np.sum(np.isnan(est_idx)) < FRACTION_TSH_RESP * len(est_idx):
            event_status_arr[idx] = "event_0"

    return event_status_arr

# UT : HIGH
def correct_right_edge_l0(event_status_arr=None, pf_event_est_arr=None,
                          offset_back_windows=None):
    """Correct right edges of event 0 (correct level to level 0)

    INPUTS:
    event_status_arr: Estimate resp event level at each time-point
    pf_event_est_arr: 1D time series of estimated P/F ratio at each time-step
    offset_back_windows: Do not process edge windows at the end of the stay

    RETURNS: Event status array with right edge of L0 zones corrected
    """
    on_right_edge = False
    in_event = False
    for idx in range(0, len(event_status_arr) - offset_back_windows):
        cur_state = event_status_arr[idx].decode()
        if cur_state in ["event_0"] and not in_event:
            in_event = True
        elif in_event and cur_state not in ["event_0"]:
            in_event = False
            on_right_edge = True
        if on_right_edge:
            if pf_event_est_arr[idx] < LEVEL1_RATIO_RESP:
                on_right_edge = False
            else:
                event_status_arr[idx] = "event_0"

    return event_status_arr

# UT : HIGH
def correct_right_edge_l1(event_status_arr=None, pf_event_est_arr=None,
                          offset_back_windows=None):
    """Correct right edges of event 1 (correct to level 1)
    INPUTS:
    event_status_arr: Estimate resp event level at each time-point
    pf_event_est_arr: 1D time series of estimated P/F ratio at each time-step
    offset_back_windows: Do not process edge windows at the end of the stay

    RETURNS: Event status array with right edge of L1 zones corrected
    """
    on_right_edge = False
    in_event = False
    for idx in range(0, len(event_status_arr) - offset_back_windows):
        cur_state = event_status_arr[idx].decode()
        if cur_state in ["event_1"] and not in_event:
            in_event = True
        elif in_event and cur_state not in ["event_1"]:
            in_event = False
            on_right_edge = True
        if on_right_edge:
            if pf_event_est_arr[idx] < LEVEL2_RATIO_RESP or pf_event_est_arr[idx] >= LEVEL1_RATIO_RESP:
                on_right_edge = False
            else:
                event_status_arr[idx] = "event_1"

    return event_status_arr

# UT : HIGH
def correct_right_edge_l2(event_status_arr=None, pf_event_est_arr=None,
                          offset_back_windows=None):
    """Correct right edges of event 2 (correct to level 2)
    INPUTS:
    event_status_arr: Estimate resp event level at each time-point
    pf_event_est_arr: 1D time series of estimated P/F ratio at each time-step
    offset_back_windows: Do not process edge windows at the end of the stay

    RETURNS: Event status array with right edge of L2 zones corrected
    """
    on_right_edge = False
    in_event = False
    for idx in range(0, len(event_status_arr) - offset_back_windows):
        cur_state = event_status_arr[idx].decode()
        if cur_state in ["event_2"] and not in_event:
            in_event = True
        elif in_event and cur_state not in ["event_2"]:
            in_event = False
            on_right_edge = True
        if on_right_edge:
            if pf_event_est_arr[idx] < LEVEL3_RATIO_RESP or pf_event_est_arr[idx] >= LEVEL2_RATIO_RESP:
                on_right_edge = False
            else:
                event_status_arr[idx] = "event_2"

    return event_status_arr

# UT : HIGH
def correct_right_edge_l3(event_status_arr=None, pf_event_est_arr=None,
                          offset_back_windows=None):
    """ Correct right edges of event 3 (correct to level 3)

    INPUTS:
    event_status_arr: Estimate resp event level at each time-point
    pf_event_est_arr: 1D time series of estimated P/F ratio at each time-step
    offset_back_windows: Do not process edge windows at the end of the stay

    RETURNS: Event status array with right edge of L3 zones corrected
    """

    on_right_edge = False
    in_event = False
    for idx in range(0, len(event_status_arr) - offset_back_windows):
        cur_state = event_status_arr[idx].decode()
        if cur_state in ["event_3"] and not in_event:
            in_event = True
        elif in_event and cur_state not in ["event_3"]:
            in_event = False
            on_right_edge = True
        if on_right_edge:
            if pf_event_est_arr[idx] >= LEVEL3_RATIO_RESP:
                on_right_edge = False
            else:
                event_status_arr[idx] = "event_3"

    return event_status_arr


def assemble_out_df(time_col=None, rel_time_col=None, pid_col=None, event_status_arr=None,
                    relabel_arr=None, fio2_avail_arr=None, fio2_suppox_arr=None,
                    fio2_ambient_arr=None, fio2_est_arr=None, pao2_est_arr=None, pao2_sur_est=None,
                    pao2_avail_arr=None, pao2_sao2_model_arr=None, pao2_full_model_arr=None,
                    ratio_arr=None, sur_ratio_arr=None, vent_status_arr=None, vent_period_arr=None,
                    vent_votes_arr=None, vent_votes_etco2_arr=None, vent_votes_ventgroup_arr=None,
                    vent_votes_tv_arr=None, vent_votes_airway_arr=None, circ_status_arr=None):
    """ Assembles the complete data-frame from the constructed endpoint and status arrays

    INPUTS:
    time_col: Absolute time column
    rel_time_col: Relative time column
    pid_col: Column with patient ID
    event_status_arr: Respiratory event levels estimated at each time-point
    relabel_arr: Binary indicator if a position was relabeled with a post-processing step, for the resp
                 status estimation.
    fio2_avail_arr: Indicator whether FiO2 measurement was available at a time-point
    fio2_suppox_arr: Indicator whether FiO2 was estimated from supplementary oxygen at a time-point
    fio2_ambient_arr: Indicator whether ambient air assumption was used at a time-point
    fio2_est_arr: FiO2 value estimated at each time-point
    pao2_est_arr: PaO2 value estimated at each time-point
    pao2_sur_est: TBD
    pao2_avail_arr: Binary indicator whether a real PaO2 measurement was available close to time-point
    pao2_sao2_model_arr: Binary indicator whether SaO2 model was used to estimate PaO2 at a time-point
    pao2_full_model_arr: Binary indicator whether full model was used to estimate PaO2 at a time-point
    ratio_arr: P/F ratios estimated at each time-point
    sur_ratio_arr: TBD
    vent_status_arr: Presence of ventilation at each time-point
    vent_period_arr: More precise post-processed version of ventilation annotation
    vent_votes_arr: Ventilation detection is based on a voting algorithm based on several factors, number
                    of points associated with each time-point
    vent_votes_etco2_arr: Ventilation votes for the ETCO2 condition
    vent_votes_ventgroup_arr: Ventilation votes for the ventilation code group condition
    vent_votes_tv_arr: Ventilation votes for the TV condition
    vent_votes_airway_arr: Ventilation votes for the airway condition
    circ_status_arr: Circulatory failure level annotated at each time-point

    RETURNS: Complete endpoint data-frame for a patient

    """
    df_out_dict = {}

    df_out_dict[DATETIME] = time_col
    df_out_dict[REL_DATETIME] = rel_time_col
    df_out_dict[PID] = pid_col
    status_list = list(map(lambda raw_str: raw_str.decode("unicode_escape"), event_status_arr.tolist()))
    df_out_dict["resp_failure_status"] = status_list
    df_out_dict["resp_failure_status_relabel"] = relabel_arr

    # Status columns
    df_out_dict["fio2_available"] = fio2_avail_arr
    df_out_dict["fio2_suppox"] = fio2_suppox_arr
    df_out_dict["fio2_ambient"] = fio2_ambient_arr
    df_out_dict["fio2_estimated"] = fio2_est_arr
    df_out_dict["pao2_estimated"] = pao2_est_arr

    df_out_dict["pao2_estimated_sur"] = pao2_sur_est
    df_out_dict["pao2_available"] = pao2_avail_arr
    df_out_dict["pao2_sao2_model"] = pao2_sao2_model_arr
    df_out_dict["pao2_full_model"] = pao2_full_model_arr
    df_out_dict["estimated_ratio"] = ratio_arr
    df_out_dict["estimated_ratio_sur"] = sur_ratio_arr
    df_out_dict["vent_state"] = vent_status_arr
    df_out_dict["vent_period"] = vent_period_arr

    # Ventilation voting base columns
    df_out_dict["vent_votes"] = vent_votes_arr
    df_out_dict["vent_votes_etco2"] = vent_votes_etco2_arr
    df_out_dict["vent_votes_ventgroup"] = vent_votes_ventgroup_arr
    df_out_dict["vent_votes_tv"] = vent_votes_tv_arr
    df_out_dict["vent_votes_airway"] = vent_votes_airway_arr

    # Circulatory failure related
    df_out_dict["circ_failure_status"] = circ_status_arr

    df_out = pd.DataFrame(df_out_dict)
    return df_out

# UT : Mid
# TODO : Refactor
def delete_short_vent_events(vent_status_arr, short_event_hours):
    """ Delete short events in the ventilation status array
    
    INPUTS:
    vent_status_arr: Original ventilation annotation with small events not yet removed
    short_event_hours: Threshold in hours when an event is considered "small"

    RETURNS: Ventilation status array with small events removed
    """
    in_event = False
    event_length = 0
    for idx in range(len(vent_status_arr)):
        cur_state = vent_status_arr[idx]
        if in_event and cur_state == 1.0:
            event_length += MINS_PER_STEP
        if not in_event and cur_state == 1.0:
            in_event = True
            event_length = MINS_PER_STEP
            event_start_idx = idx
        if in_event and (cur_state == 0.0 or np.isnan(cur_state)):
            in_event = False
            if event_length / 60. < short_event_hours:
                vent_status_arr[event_start_idx:idx] = 0.0
    return vent_status_arr

# UT : HIGH
# TODO Check there is no hardcoding here
def ellis(x_orig):
    """ ELLIS model converting SpO2 in 100 % units into a PaO2 ABGA
        estimate
    
    INPUTS: 
    x_orig: SpO2 values from which to estimate

    RETURNS: Estimated PaO2 values for each time-step
    """
    x_orig[np.isnan(x_orig)] = SPO2_NORMAL_VALUE  # Normal value assumption
    x = x_orig / 100
    x[x == 1] = 0.999
    exp_base = (11700 / ((1 / x) - 1))
    exp_sqrbracket = np.sqrt(pow(50, 3) + (exp_base ** 2))
    exp_first = np.cbrt(exp_base + exp_sqrbracket)
    exp_second = np.cbrt(exp_base - exp_sqrbracket)
    exp_full = exp_first + exp_second
    return exp_full

# UT : HIGH
def correct_left_edge_vent(vent_status_arr, etco2_meas_cnt, etco2_col):
    """ Corrects the left edge of the ventilation status array, to pin-point the exact conditions

    INPUTS:
    vent_status_arr: Ventilation detection before edge correction took place
    etco2_meas_cnt: Cumulative number of ETCO2 measurements at each time-step since beginning of stay

    RETURNS: Ventilation annotation with left edge of events corrected
    """ 
    on_left_edge = False
    in_event = False

    # Correct left ventilation edges of the ventilation zone
    for idx in range(len(vent_status_arr)):
        if vent_status_arr[idx] == 1.0 and not in_event:
            in_event = True
            on_left_edge = True
        if on_left_edge and in_event:
            if vent_status_arr[idx] == 0.0:
                in_event = False
                on_left_edge = False
            elif (idx == 0 and etco2_meas_cnt[idx] > 0 or etco2_meas_cnt[idx] - etco2_meas_cnt[idx - 1] >= 1) and \
                    etco2_col[idx] > VENT_ETCO2_TSH:
                on_left_edge = False
            else:
                vent_status_arr[idx] = 0.0

    return vent_status_arr


# UT : HIGH
def delete_small_continuous_blocks(event_arr, block_threshold=None):
    """ Given an event array, deletes small contiguous blocks that are sandwiched between two other blocks, one of which
        is longer, they both have the same label. For the moment we delete blocks smaller than 30 minutes. Note this
        requires only a linear pass over the array

    INPUTS:
    event_arr: Binary event indication array (0 no event, 1 event)
    block_threshold: Blocks smaller than this length should be removed

    RETURNS: Event array after small continuous blocks are removed

    """
    block_list = []
    active_block = None

    # Build a block list
    for jdx in range(event_arr.size):
        new_block = event_arr[jdx]

        # Start a new block at the beginning
        if active_block is None:
            active_block = new_block
            left_block_idx = jdx

        # Change to a new block
        elif not active_block == new_block:
            block_list.append((active_block, left_block_idx, jdx - 1))
            left_block_idx = jdx
            active_block = new_block

        # Same last block unconditionally
        if jdx == event_arr.size - 1:
            block_list.append((new_block, left_block_idx, jdx))

    # Merge blocks
    while True:
        all_clean = True
        for bidx, block in enumerate(block_list):
            block_label, lidx, ridx = block
            block_len = ridx - lidx + 1

            # Candidate for merging
            if block_len <= block_threshold:

                if len(block_list) == 1:
                    all_clean = True
                    break

                # Only right block
                elif bidx == 0:
                    next_block = block_list[bidx + 1]
                    nb_label, nb_lidx, nb_ridx = next_block
                    nb_len = nb_ridx - nb_lidx + 1

                    # Merge blocks
                    if nb_len > block_len and nb_len > block_threshold:
                        block_list[bidx] = (nb_label, lidx, nb_ridx)
                        block_list.remove(next_block)
                        all_clean = False
                        break

                # Only left block
                elif bidx == len(block_list) - 1:
                    prev_block = block_list[bidx - 1]
                    pb_label, pb_lidx, pb_ridx = prev_block
                    pb_len = pb_ridx - pb_lidx + 1

                    if pb_len > block_len and pb_len > block_threshold:
                        block_list[bidx] = (pb_label, pb_lidx, ridx)
                        block_list.remove(prev_block)
                        all_clean = False
                        break

                # Interior block
                else:
                    prev_block = block_list[bidx - 1]
                    next_block = block_list[bidx + 1]
                    pb_label, pb_lidx, pb_ridx = prev_block
                    nb_label, nb_lidx, nb_ridx = next_block
                    pb_len = pb_ridx - pb_lidx + 1
                    nb_len = nb_ridx - nb_lidx + 1

                    if pb_label == nb_label and (pb_len > block_threshold or nb_len > block_threshold):
                        block_list[bidx] = (pb_label, pb_lidx, nb_ridx)
                        block_list.remove(prev_block)
                        block_list.remove(next_block)
                        all_clean = False
                        break

        # Traversed block list with no actions required
        if all_clean:
            break

    # Now back-translate the block list to the list
    out_arr = np.copy(event_arr)

    for blabel, lidx, ridx in block_list:
        out_arr[lidx:ridx + 1] = blabel

    # Additionally build an array where the two arrays are different
    diff_arr = (out_arr != event_arr).astype(np.bool)

    return (out_arr, diff_arr)

# UT : Mid
def gen_circ_failure_ep(event_status_arr=None, map_col=None, lactate_col=None, milri_col=None, dobut_col=None,
                        levosi_col=None, theo_col=None,
                        noreph_col=None, epineph_col=None, vaso_col=None):
    """ Circulatory failure endpoint definition"""
    circ_status_arr = np.zeros_like(map_col)

    # Computation of the circulatory failure toy version of the endpoint
    for jdx in range(0, len(event_status_arr)):
        map_subarr = map_col[max(0, jdx - STEPS_PER_HOUR):min(jdx + STEPS_PER_HOUR, len(event_status_arr))]
        lact_subarr = lactate_col[max(0, jdx - STEPS_PER_HOUR):min(jdx + STEPS_PER_HOUR, len(event_status_arr))]
        milri_subarr = milri_col[max(0, jdx - STEPS_PER_HOUR):min(jdx + STEPS_PER_HOUR, len(event_status_arr))]
        dobut_subarr = dobut_col[max(0, jdx - STEPS_PER_HOUR):min(jdx + STEPS_PER_HOUR, len(event_status_arr))]
        levosi_subarr = levosi_col[max(0, jdx - STEPS_PER_HOUR):min(jdx + STEPS_PER_HOUR, len(event_status_arr))]
        theo_subarr = theo_col[max(0, jdx - STEPS_PER_HOUR):min(jdx + STEPS_PER_HOUR, len(event_status_arr))]
        noreph_subarr = noreph_col[max(0, jdx - STEPS_PER_HOUR):min(jdx + STEPS_PER_HOUR, len(event_status_arr))]
        epineph_subarr = epineph_col[max(0, jdx - STEPS_PER_HOUR):min(jdx + STEPS_PER_HOUR, len(event_status_arr))]
        vaso_subarr = vaso_col[max(0, jdx - STEPS_PER_HOUR):min(jdx + STEPS_PER_HOUR, len(event_status_arr))]
        map_crit_arr = ((map_subarr < 65) | (milri_subarr > 0) | (dobut_subarr > 0) | (levosi_subarr > 0) | (
                    theo_subarr > 0) | (noreph_subarr > 0) | (epineph_subarr > 0) | (vaso_subarr > 0))
        lact_crit_arr = (lact_subarr > 2)
        map_condition = np.sum(map_crit_arr) >= FRACTION_TSH_CIRC * len(map_crit_arr)
        lact_condition = np.sum(lact_crit_arr) >= FRACTION_TSH_CIRC * len(map_crit_arr)
        if map_condition and lact_condition:
            circ_status_arr[jdx] = 1.0

    return circ_status_arr

# UT : V HIGH
def compute_pao2_fio2_estimates(ratio_arr=None, abs_dtime_arr=None, suppox_async_red_ts=None,
                                fio2_col=None, fio2_meas_cnt=None,
                                pao2_meas_cnt=None, pao2_col=None, spo2_col=None,
                                abga_window=None, spo2_meas_cnt=None, pao2_est_arr=None, fio2_est_arr=None,
                                vent_mode_col=None, vent_status_arr=None, event_count=None,
                                fio2_avail_arr=None, suppox_max_ffill=None, ambient_fio2=None,
                                fio2_ambient_arr=None, suppox_val=None, fio2_suppox_arr=None,
                                sz_fio2_window=None, sz_pao2_window=None, pao2_avail_col=None):

    # Array pointers tracking the current active value of each type
    suppox_async_red_ptr = -1

    # Label each point in the 30 minute window (except ventilation)
    for jdx in range(0, len(ratio_arr)):

        # Advance to the last SuppOx infos before grid point
        cur_time = abs_dtime_arr[jdx]
        while True:
            suppox_async_red_ptr = suppox_async_red_ptr + 1
            if suppox_async_red_ptr >= len(suppox_async_red_ts) or suppox_async_red_ts[
                suppox_async_red_ptr] > cur_time:
                suppox_async_red_ptr = suppox_async_red_ptr - 1
                break

        # Estimate the current FiO2 value
        bw_fio2 = fio2_col[max(0, jdx - sz_fio2_window):jdx + 1]
        bw_fio2_meas = fio2_meas_cnt[max(0, jdx - sz_fio2_window):jdx + 1]
        fio2_meas = bw_fio2_meas[-1] - bw_fio2_meas[0] > 0

        mode_group_est = vent_mode_col[jdx]

        # FiO2 is measured since beginning of stay and EtCO2 was measured, we use FiO2 (indefinite forward filling)
        # if ventilation is active or the current estimate of ventilation mode group is NIV.
        if fio2_meas and (vent_status_arr[jdx] == 1.0 or mode_group_est == NIV_VENT_MODE):
            event_count["FIO2_AVAILABLE"] += 1
            fio2_val = bw_fio2[-1] / 100
            fio2_avail_arr[jdx] = 1

        # Use supplemental oxygen or ambient air oxygen
        else:

            # No real measurements up to now, or the last real measurement
            # was more than 8 hours away.
            if suppox_async_red_ptr == -1 or (
                    cur_time - suppox_async_red_ts[suppox_async_red_ptr]) > np.timedelta64(suppox_max_ffill, 'h'):
                event_count["SUPPOX_NO_MEAS_12_HOURS_LIMIT"] += 1
                fio2_val = ambient_fio2
                fio2_ambient_arr[jdx] = 1

            # Find the most recent source variable of SuppOx
            else:
                suppox = suppox_val["SUPPOX"][suppox_async_red_ptr]

                # SuppOx information from main source
                if np.isfinite(suppox):
                    event_count["SUPPOX_MAIN_VAR"] += 1
                    fio2_val = suppox_to_fio2(int(suppox)) / 100
                    fio2_suppox_arr[jdx] = 1
                else:
                    raise Exception("Impossible condition for suppox and fio2")

        bw_pao2_meas = pao2_meas_cnt[max(0, jdx - sz_pao2_window):jdx + 1]
        bw_pao2 = pao2_col[max(0, jdx - sz_pao2_window):jdx + 1]
        pao2_meas = bw_pao2_meas[-1] - bw_pao2_meas[0] >= 1

        # PaO2 was just measured, just use the value
        if pao2_meas:
            pao2_estimate = bw_pao2[-1]
            pao2_avail_col[jdx] = 1

        # Have to forecast PaO2 from a previous SpO2
        else:
            bw_spo2 = spo2_col[max(0, jdx - abga_window):jdx + 1]
            bw_spo2_meas = spo2_meas_cnt[max(0, jdx - abga_window):jdx + 1]
            spo2_meas = bw_spo2_meas[-1] - bw_spo2_meas[0] >= 1

            # Standard case, take the last SpO2 measurement
            if spo2_meas:
                spo2_val = bw_spo2[-1]
                pao2_estimate = ellis(np.array([spo2_val]))[0]

            # Extreme edge case, there was SpO2 measurement in the last 24 hours
            else:
                spo2_val = SPO2_NORMAL_VALUE
                pao2_estimate = ellis(np.array([spo2_val]))[0]

        # Compute the individual components of the Horowitz index
        pao2_est_arr[jdx] = pao2_estimate
        fio2_est_arr[jdx] = fio2_val

        out_dict = {}
        out_dict["pao2_est"] = pao2_est_arr
        out_dict["fio2_est"] = fio2_est_arr
        out_dict["fio2_avail"] = fio2_avail_arr
        out_dict["fio2_suppox"] = fio2_suppox_arr
        out_dict["fio2_ambient"] = fio2_ambient_arr

        return out_dict

# UT : Mid
def delete_low_density_hr_gap(vent_status_arr, hr_status_arr, configs=None):
    """ Deletes gaps in ventilation which are caused by likely sensor dis-connections"""
    in_event = False
    in_gap = False
    gap_idx = -1
    for idx in range(len(vent_status_arr)):

        # Beginning of new event, not from inside gap
        if not in_event and not in_gap and vent_status_arr[idx] == 1.0:
            in_event = True

        # Beginning of potential gap that needs to be closed
        elif in_event and vent_status_arr[idx] == 0.0:
            in_gap = True
            gap_idx = idx
            in_event = False

        # The gap is over, re-assign the status of ventilation to merge the gap, enter new event
        if in_gap and vent_status_arr[idx] == 1.0:

            hr_sub_arr = hr_status_arr[gap_idx:idx]

            # Close the gap if the density of HR is too low in between
            if np.sum(hr_sub_arr) / hr_sub_arr.size <= configs["vent_hr_density_threshold"]:
                vent_status_arr[gap_idx:idx] = 1.0

            in_gap = False
            in_event = True

    return vent_status_arr


def load_relevant_columns(df_pid, var_map):
    """ Loads the relevant columns from the patient stay"""
    pat_cols = {}

    pat_cols["fio2"] = np.array(df_pid[var_map["FiO2"]])
    pat_cols["pao2"] = np.array(df_pid[var_map["PaO2"]])
    pat_cols["etco2"] = np.array(df_pid[var_map["etCO2"]])

    pat_cols["noreph"] = np.array(df_pid[var_map["Norephenephrine"][0]])
    pat_cols["epineph"] = np.array(df_pid[var_map["Epinephrine"][0]])
    pat_cols["vaso"] = np.array(df_pid[var_map["Vasopressin"][0]])

    pat_cols["milri"] = np.array(df_pid[var_map["Milrinone"][0]])
    pat_cols["dobut"] = np.array(df_pid[var_map["Dobutamine"][0]])
    pat_cols["levosi"] = np.array(df_pid[var_map["Levosimendan"][0]])
    pat_cols["theo"] = np.array(df_pid[var_map["Theophyllin"][0]])

    pat_cols["lactate"] = np.array(df_pid[var_map["Lactate"][0]])
    pat_cols["peep"] = np.array(df_pid[var_map["PEEP"]])

    # Heartrate
    pat_cols["hr_meas_cnt"] = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["HR"])])

    pat_cols["tv"] = np.array(df_pid[var_map["TV"]])
    pat_cols["map"] = np.array(df_pid[var_map["MAP"][0]])
    pat_cols["airway"] = np.array(df_pid[var_map["Airway"]])

    # Ventilator mode group columns
    pat_cols["vent_mode"] = np.array(df_pid[var_map["vent_mode"]])

    pat_cols["spo2"] = np.array(df_pid[var_map["SpO2"]])

    pat_cols["fio2_meas_cnt"] = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["FiO2"])])
    pat_cols["pao2_meas_cnt"] = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["PaO2"])])
    pat_cols["etco2_meas_cnt"] = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["etCO2"])])
    pat_cols["peep_meas_cnt"] = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["PEEP"])])
    pat_cols["hr_meas_cnt"] = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["HR"])])
    pat_cols["spo2_meas_cnt"] = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["SpO2"])])

    # Absolute time
    pat_cols["abs_dtime"] = np.array(df_pid["datetime"])

    return pat_cols


def initialize_status_cols(fio2_col=None):
    stat_arr = {}

    event_status_arr = np.zeros(fio2_col.size, dtype="<S10")
    event_status_arr.fill("UNKNOWN")
    stat_arr["event_status"] = event_status_arr

    # Status arrays
    stat_arr["pao2_avail"] = np.zeros(fio2_col.size)
    stat_arr["fio2_avail"] = np.zeros(fio2_col.size)
    stat_arr["fio2_suppox"] = np.zeros(fio2_col.size)
    stat_arr["fio2_ambient"] = np.zeros(fio2_col.size)
    stat_arr["pao2_sao2_model"] = np.zeros(fio2_col.size)
    stat_arr["pao2_full_model"] = np.zeros(fio2_col.size)

    stat_arr["ratio"] = np.zeros(fio2_col.size)
    stat_arr["sur_ratio"] = np.zeros(fio2_col.size)

    stat_arr["pao2_est"] = np.zeros(fio2_col.size)
    stat_arr["fio2_est"] = np.zeros(fio2_col.size)
    stat_arr["vent_status"] = np.zeros(fio2_col.size)

    readiness_ext_arr = np.zeros(fio2_col.size)
    readiness_ext_arr[:] = np.nan
    stat_arr["readiness_ext"] = readiness_ext_arr

    # Votes arrays
    stat_arr["vent_votes"] = np.zeros(fio2_col.size)
    stat_arr["vent_votes_etco2"] = np.zeros(fio2_col.size)
    stat_arr["vent_votes_ventgroup"] = np.zeros(fio2_col.size)
    stat_arr["vent_votes_tv"] = np.zeros(fio2_col.size)
    stat_arr["vent_votes_airway"] = np.zeros(fio2_col.size)

    stat_arr["peep_status"] = np.zeros(fio2_col.size)
    stat_arr["peep_threshold"] = np.zeros(fio2_col.size)
    stat_arr["hr_status"] = np.zeros(fio2_col.size)
    stat_arr["etco2_status"] = np.zeros(fio2_col.size)

    return stat_arr


def suppox_to_fio2(suppox_val):
    """ Conversion of supplemental oxygen to FiO2 estimated value"""
    if suppox_val > MAX_SUPPOX_KEY:
        return MAX_SUPPOX_TO_FIO2_VAL
    else:
        return SUPPOX_TO_FIO2[suppox_val]


def endpoint_gen_benchmark(configs):
    var_map = configs["VAR_IDS"]
    sz_window = configs["length_fw_window"]
    abga_window = configs["length_ABGA_window"]

    imputed_f = configs["imputed_path"]
    merged_f = os.path.join(configs["merged_h5"])
    out_folder = os.path.join(configs["endpoint_path"])

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    batch_id = configs["batch_idx"]

    logging.info("Generating endpoints for batch {}".format(batch_id))
    batch_fpath = os.path.join(imputed_f, "batch_{}.parquet".format(batch_id))

    if not os.path.exists(batch_fpath):
        logging.info("WARNING: Input file does not exist, exiting...")
        sys.exit(1)

    if configs["load_batch_at_once"]:
        df_batch = pd.read_parquet(os.path.join(imputed_f, "batch_{}.parquet".format(batch_id)))

    logging.info("Loaded imputed data done...")
    cand_raw_batch = glob.glob(os.path.join(merged_f, "part-{}.parquet".format(batch_id)))
    assert (len(cand_raw_batch) == 1)
    pids = list(df_batch.patientid.unique())

    logging.info("Number of patients in batch: {}".format(len(df_batch.patientid.unique())))
    out_fp = os.path.join(out_folder, "batch_{}.parquet".format(batch_id))

    event_count = {"FIO2_AVAILABLE": 0, "SUPPOX_NO_MEAS_12_HOURS_LIMIT": 0, "SUPPOX_MAIN_VAR": 0, "SUPPOX_HIGH_FLOW": 0,
                   "SUPPOX_NO_FILL_STOP": 0}

    out_dfs = []

    for pidx, pid in enumerate(pids):

        print("Patient {}/{}".format(pidx+1,len(pids)))

        if configs["load_batch_at_once"]:
            df_pid = df_batch[df_batch["patientid"] == pid]
        else:
            df_pid = pd.read_parquet(os.path.join(imputed_f, "batch_{}.parquet".format(batch_id)),
                                     filters=[("patientid", "=", pid)])

        if df_pid.shape[0] == 0:
            logging.info("WARNING: No input data for PID: {}".format(pid))
            continue

        df_merged_pid = pd.read_parquet(cand_raw_batch[0], filters=[("patientid", "=", pid)])
        df_merged_pid.sort_values(by="datetime", inplace=True)

        suppox_val = {}

        # Main route of SuppOx
        df_suppox_red_async = df_merged_pid[[var_map["SuppOx"], "datetime"]]
        df_suppox_red_async = df_suppox_red_async.dropna(how="all", thresh=2)
        suppox_async_red_ts = np.array(df_suppox_red_async["datetime"])

        suppox_val["SUPPOX"] = np.array(df_suppox_red_async[var_map["SuppOx"]])

        # Strategy is to create an imputed SuppOx column based on the spec using
        # forward filling heuristics

        # Load patient columns from data-frame
        pat_cols = load_relevant_columns(df_pid, var_map)

        if configs["presmooth_spo2"]:
            spo2_col = percentile_smooth(pat_cols["spo2"], configs["spo2_smooth_percentile"],
                                         configs["spo2_smooth_window_size_mins"])
        else:
            spo2_col = pat_cols["spo2"]

        # Initialize status columns for this patient
        status_cols = initialize_status_cols(fio2_col=pat_cols["fio2"])

        # ======================== VENTILATION =========================================================================

        # Label each point in the 30 minute window with ventilation

        for jdx in range(0, len(status_cols["ratio"])):
            low_vent_idx = max(0, jdx - configs["peep_search_bw"])
            high_vent_idx = min(len(status_cols["ratio"]), jdx + configs["peep_search_bw"])
            low_peep_idx = max(0, jdx - configs["peep_search_bw"])
            high_peep_idx = min(len(status_cols["ratio"]), jdx + configs["peep_search_bw"])
            low_hr_idx = max(0, jdx - configs["hr_vent_search_bw"])
            high_hr_idx = min(len(status_cols["ratio"]), jdx + configs["hr_vent_search_bw"])

            win_etco2 = pat_cols["etco2"][low_vent_idx:high_vent_idx]
            win_etco2_meas = pat_cols["etco2_meas_cnt"][low_vent_idx:high_vent_idx]
            win_peep = pat_cols["peep"][low_peep_idx:high_peep_idx]
            win_peep_meas = pat_cols["peep_meas_cnt"][low_peep_idx:high_peep_idx]
            win_hr_meas = pat_cols["hr_meas_cnt"][low_hr_idx:high_hr_idx]

            etco2_meas_win = win_etco2_meas[-1] - win_etco2_meas[0] > 0
            peep_meas_win = win_peep_meas[-1] - win_peep_meas[0] > 0
            hr_meas_win = win_hr_meas[-1] - win_hr_meas[0] > 0
            current_vent_group = pat_cols["vent_mode"][jdx]
            current_tv = pat_cols["tv"][jdx]
            current_airway = pat_cols["airway"][jdx]

            vote_score = 0

            # EtCO2 requirement (still needed)
            if etco2_meas_win and (win_etco2 > 0.5).any():
                vote_score += 2
                status_cols["vent_votes_etco2"][jdx] = 2

            # Ventilation group requirement (still needed)
            if current_vent_group in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
                vote_score += 1
                status_cols["vent_votes_ventgroup"][jdx] += 1
            elif current_vent_group in [1.0]:
                vote_score -= 1
                status_cols["vent_votes_ventgroup"][jdx] -= 1
            elif current_vent_group in [11.0, 12.0, 13.0, 15.0, 17.0]:
                vote_score -= 2
                status_cols["vent_votes_ventgroup"][jdx] -= 2

            # TV presence requirement (still needed)
            if current_tv > 0:
                vote_score += 1
                status_cols["vent_votes_tv"][jdx] = 1

            # Airway requirement (still needed)
            if current_airway in [1, 2]:
                vote_score += 2
                status_cols["vent_votes_airway"][jdx] = 2

            # No airway (still needed)
            if current_airway in [3, 4, 5, 6]:
                vote_score -= 1
                status_cols["vent_votes_airway"][jdx] = -1

            status_cols["vent_votes"][jdx] = vote_score

            if vote_score >= configs["vent_vote_threshold"]:
                status_cols["vent_status"][jdx] = 1

            if peep_meas_win:
                status_cols["peep_status"][jdx] = 1
            if (win_peep >= configs["peep_threshold"]).any():
                status_cols["peep_threshold"][jdx] = 1
            if etco2_meas_win:
                status_cols["etco2_status"][jdx] = 1
            if hr_meas_win:
                status_cols["hr_status"][jdx] = 1

        if configs["detect_hr_gaps"]:
            vent_status_arr = delete_low_density_hr_gap(status_cols["vent_status"], status_cols["hr_status"],
                                                        configs=configs)

        if configs["merge_short_vent_gaps"]:
            vent_status_arr = merge_short_vent_gaps(vent_status_arr, configs["short_gap_hours"])

        if configs["delete_short_vent_events"]:
            vent_status_arr = delete_short_vent_events(vent_status_arr, configs["short_event_hours"])

        # vent_status_arr=correct_left_edge_vent(vent_status_arr, etco2_meas_cnt, etco2_col)
        # vent_status_arr=correct_right_edge_vent(vent_status_arr, etco2_meas_cnt, etco2_col)

        # Ventilation period array
        vent_period_arr = np.copy(vent_status_arr)

        # Delete short ventilation periods if no HR gap before
        in_event = False
        event_length = 0
        for idx in range(len(vent_period_arr)):
            cur_state = vent_period_arr[idx]
            if in_event and cur_state == 1.0:
                event_length += MINS_PER_STEP
            if not in_event and cur_state == 1.0:
                in_event = True
                event_length = MINS_PER_STEP
                event_start_idx = idx
            if in_event and (np.isnan(cur_state) or cur_state == 0.0):
                in_event = False

                # Short event at beginning of stay shall never be deleted...
                if event_start_idx == 0:
                    delete_event = False
                else:
                    search_hr_idx = event_start_idx - 1
                    while search_hr_idx >= 0:
                        if status_cols["hr_status"][search_hr_idx] == 1.0:
                            hr_gap_length = MINS_PER_STEP * (event_start_idx - search_hr_idx)
                            delete_event = True
                            break
                        search_hr_idx -= 1

                    # Found no HR before event, do not delete event...
                    if search_hr_idx == -1:
                        delete_event = False

                # Delete event in principle, then check if short enough...
                if delete_event:
                    event_length += hr_gap_length
                    if event_length / 60. <= configs["short_event_hours_vent_period"]:
                        vent_period_arr[event_start_idx:idx] = 0.0

        # Estimate the FiO2/PaO2 values at indiviual time points
        est_out_dict = compute_pao2_fio2_estimates(ratio_arr=status_cols["ratio"], abs_dtime_arr=pat_cols["abs_dtime"],
                                                   suppox_async_red_ts=suppox_async_red_ts, abga_window=abga_window,
                                                   fio2_col=pat_cols["fio2"], pao2_col=pat_cols["pao2"],
                                                   spo2_col=spo2_col,
                                                   fio2_meas_cnt=pat_cols["fio2_meas_cnt"],
                                                   pao2_meas_cnt=pat_cols["pao2_meas_cnt"],
                                                   spo2_meas_cnt=pat_cols["spo2_meas_cnt"],
                                                   vent_mode_col=pat_cols["vent_mode"], vent_status_arr=vent_status_arr,
                                                   event_count=event_count, fio2_avail_arr=status_cols["fio2_avail"],
                                                   suppox_max_ffill=configs["suppox_max_ffill"],
                                                   pao2_est_arr=status_cols["pao2_est"],
                                                   fio2_est_arr=status_cols["fio2_est"],
                                                   ambient_fio2=configs["ambient_fio2"],
                                                   fio2_ambient_arr=status_cols["fio2_ambient"],
                                                   suppox_val=suppox_val, fio2_suppox_arr=status_cols["fio2_suppox"],
                                                   sz_fio2_window=configs["sz_fio2_window"],
                                                   sz_pao2_window=configs["sz_pao2_window"])

        pao2_est_arr = est_out_dict["pao2_est"]
        fio2_est_arr = est_out_dict["fio2_est"]
        fio2_avail_arr = est_out_dict["fio2_avail"]
        fio2_suppox_arr = est_out_dict["fio2_suppox"]
        fio2_ambient_arr = est_out_dict["fio2_ambient"]

        # Smooth individual components of the P/F ratio estimate
        if configs["kernel_smooth_estimate_pao2"]:
            pao2_est_arr = kernel_smooth_arr(est_out_dict["pao2_est"], bandwidth=configs["smoothing_bandwidth"])

        # Convex combination of the estimate
        if configs["mix_real_estimated_pao2"]:
            pao2_est_arr = mix_real_est_pao2(pat_cols["pao2"], pat_cols["pao2_meas_cnt"], pao2_est_arr)

        if configs["kernel_smooth_estimate_fio2"]:
            fio2_est_arr = kernel_smooth_arr(est_out_dict["fio2_est"], bandwidth=configs["smoothing_bandwidth"])

        # Test2 data-set for surrogate model
        pao2_sur_est = np.copy(pao2_est_arr)
        assert (np.sum(np.isnan(pao2_sur_est)) == 0)

        ratio_arr = np.divide(pao2_est_arr, fio2_est_arr)

        # Post-smooth Horowitz index
        if configs["post_smooth_pf_ratio"]:
            ratio_arr = kernel_smooth_arr(ratio_arr, bandwidth=configs["post_smoothing_bandwidth"])

        if configs["pao2_version"] == "ellis_basic":
            pf_event_est_arr = np.copy(ratio_arr)
        elif configs["pao2_version"] == "original":
            assert (False)

        event_status_arr = assign_resp_levels(event_status_arr=status_cols["event_status"],
                                              pf_event_est_arr=pf_event_est_arr,
                                              vent_status_arr=vent_status_arr,
                                              peep_status_arr=status_cols["peep_status"],
                                              ratio_arr=ratio_arr, sz_window=sz_window,
                                              peep_threshold_arr=status_cols["peep_threshold"],
                                              offset_back_windows=configs["offset_back_windows"])

        # Re-traverse the array and correct the right edges of events
        event_status_arr = correct_right_edge_l0(event_status_arr=event_status_arr, pf_event_est_arr=pf_event_est_arr,
                                                 offset_back_windows=configs["offset_back_windows"])
        event_status_arr = correct_right_edge_l1(event_status_arr=event_status_arr, pf_event_est_arr=pf_event_est_arr,
                                                 offset_back_windows=configs["offset_back_windows"])
        event_status_arr = correct_right_edge_l2(event_status_arr=event_status_arr, pf_event_est_arr=pf_event_est_arr,
                                                 offset_back_windows=configs["offset_back_windows"])
        event_status_arr = correct_right_edge_l3(event_status_arr=event_status_arr, pf_event_est_arr=pf_event_est_arr,
                                                 offset_back_windows=configs["offset_back_windows"])

        circ_status_arr = gen_circ_failure_ep(event_status_arr=event_status_arr,
                                              map_col=pat_cols["map"], lactate_col=pat_cols["lactate"],
                                              milri_col=pat_cols["milri"], dobut_col=pat_cols["dobut"],
                                              levosi_col=pat_cols["levosi"], theo_col=pat_cols["theo"],
                                              noreph_col=pat_cols["noreph"], epineph_col=pat_cols["epineph"],
                                              vaso_col=pat_cols["vaso"])

        # Traverse the array and delete short gap
        event_status_arr, relabel_arr = delete_small_continuous_blocks(event_status_arr,
                                                                       block_threshold=configs[
                                                                           "pf_event_merge_threshold"])

        time_col = np.array(df_pid["datetime"])
        rel_time_col = np.array(df_pid["rel_datetime"])
        pid_col = np.array(df_pid["patientid"])

        df_out = assemble_out_df(time_col=time_col, rel_time_col=rel_time_col, pid_col=pid_col,
                                 event_status_arr=event_status_arr,
                                 relabel_arr=relabel_arr, fio2_avail_arr=fio2_avail_arr,
                                 fio2_suppox_arr=fio2_suppox_arr,
                                 fio2_ambient_arr=fio2_ambient_arr, fio2_est_arr=fio2_est_arr,
                                 pao2_est_arr=pao2_est_arr, pao2_sur_est=pao2_sur_est,
                                 pao2_avail_arr=status_cols["pao2_avail"],
                                 pao2_sao2_model_arr=status_cols["pao2_sao2_model"],
                                 pao2_full_model_arr=status_cols["pao2_full_model"],
                                 ratio_arr=ratio_arr, sur_ratio_arr=status_cols["sur_ratio"],
                                 vent_status_arr=vent_status_arr, vent_period_arr=vent_period_arr,
                                 vent_votes_arr=status_cols["vent_votes"],
                                 vent_votes_etco2_arr=status_cols["vent_votes_etco2"],
                                 vent_votes_ventgroup_arr=status_cols["vent_votes_ventgroup"],
                                 vent_votes_tv_arr=status_cols["vent_votes_tv"],
                                 vent_votes_airway_arr=status_cols["vent_votes_airway"],
                                 circ_status_arr=circ_status_arr)

        out_dfs.append(df_out)

    all_df = pd.concat(out_dfs, axis=0)
    all_df.to_parquet(out_fp)
