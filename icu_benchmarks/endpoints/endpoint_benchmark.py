""" Functions to generate the set of endpoints for the time series
    benchmark on the HiRID database"""

import glob
import logging
import math
import os
import os.path
import pickle
import random
import sys

import lightgbm as lgbm
import numpy as np
import pandas as pd
import skfda.preprocessing.smoothing.kernel_smoothers as skks
import skfda.representation.grid as skgrid
import sklearn.linear_model as sklm
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpproc


def load_pickle(fpath):
    """ Given a file path pointing to a pickle file, yields the object pickled in this file"""
    with open(fpath, 'rb') as fp:
        return pickle.load(fp)


SUPPOX_TO_FIO2 = {
    0: 21,
    1: 26,
    2: 34,
    3: 39,
    4: 45,
    5: 49,
    6: 54,
    7: 57,
    8: 58,
    9: 63,
    10: 66,
    11: 67,
    12: 69,
    13: 70,
    14: 73,
    15: 75}


def mix_real_est_pao2(pao2_col, pao2_meas_cnt, pao2_est_arr, bandwidth=None):
    """ Mix real PaO2 measurement and PaO2 estimates using a Gaussian kernel"""
    final_pao2_arr = np.copy(pao2_est_arr)
    sq_scale = 57 ** 2  # 1 hour has mass 1/3 approximately

    for idx in range(final_pao2_arr.size):
        meas_ref = pao2_meas_cnt[idx]
        real_val = None
        real_val_dist = None

        # Search forward and backward with priority giving to backward if equi-distant
        for sidx in range(48):
            if not idx - sidx < 0 and pao2_meas_cnt[idx - sidx] < meas_ref:
                real_val = pao2_col[idx - sidx + 1]
                real_val_dist = 5 * sidx
                break
            elif not idx + sidx >= final_pao2_arr.size and pao2_meas_cnt[idx + sidx] > meas_ref:
                real_val = pao2_col[idx + sidx]
                real_val_dist = 5 * sidx
                break

        if real_val is not None:
            alpha_mj = math.exp(-real_val_dist ** 2 / sq_scale)
            alpha_ej = 1 - alpha_mj
            final_pao2_arr[idx] = alpha_mj * real_val + alpha_ej * pao2_est_arr[idx]

    return final_pao2_arr


def perf_regression_model(X_list, y_list, aux_list, configs=None):
    """ Initial test of a regression model to estimate the current Pao2 based
        on 6 features of the past. Also pass FiO2 to calculate resulting mistakes in
        the P/F ratio"""

    logging.info("Testing regression model for PaO2...")

    # Partition the data into 3 sets and run SGD regressor
    X_train = X_list[:int(0.6 * len(X_list))]
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_list[:int(0.6 * len(y_list))])
    X_val = X_list[int(0.6 * len(X_list)):int(0.8 * len(X_list))]
    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_list[int(0.6 * len(y_list)):int(0.8 * len(y_list))])
    X_test = X_list[int(0.8 * len(X_list)):]
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_list[int(0.8 * len(y_list)):])

    fio2_test = np.concatenate(aux_list[int(0.8 * len(aux_list)):])

    if configs["sur_model_type"] == "linear":
        scaler = skpproc.StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_val_std = scaler.transform(X_val)
        X_test_std = scaler.transform(X_test)

    if configs["sur_model_type"] == "linear":
        alpha_cands = [0.0001, 0.001, 0.01, 0.1, 1.0]
    elif configs["sur_model_type"] == "lgbm":
        alpha_cands = [32]

    best_alpha = None
    best_score = np.inf

    # Search for the best model on the validation set
    for alpha in alpha_cands:
        logging.info("Testing alpha: {}".format(alpha))

        if configs["sur_model_type"] == "linear":
            lmodel_cand = sklm.SGDRegressor(alpha=alpha, random_state=2021)
        elif configs["sur_model_type"] == "lgbm":
            lmodel_cand = lgbm.LGBMRegressor(num_leaves=alpha, learning_rate=0.05, n_estimators=1000,
                                             random_state=2021)

        if configs["sur_model_type"] == "linear":
            lmodel_cand.fit(X_train_std, y_train)
        elif configs["sur_model_type"] == "lgbm":
            lmodel_cand.fit(X_train_std, y_train, eval_set=(X_val_std, y_val), early_stopping_rounds=20,
                            eval_metric="mae")

        pred_y_val = lmodel_cand.predict(X_val_std)
        mae_val = np.median(np.absolute(y_val - pred_y_val))
        if mae_val < best_score:
            best_score = mae_val
            best_alpha = alpha

    lmodel = sklm.SGDRegressor(alpha=best_alpha, random_state=2021)
    lmodel.fit(X_train_std, y_train)
    pred_y_test = lmodel.predict(X_test_std)

    # MH Never used
    pred_pf_ratio_test = pred_y_test / fio2_test
    true_pf_ratio_test = y_test / fio2_test

    mae_test = skmetrics.mean_absolute_error(y_test, pred_y_test)
    logging.info("Mean absolute error in test set: {:.3f}".format(mae_test))


def percentile_smooth(signal_col, percentile, win_scope_mins):
    """ Window percentile smoother, where percentile is in the interval [0,100]"""
    out_arr = np.zeros_like(signal_col)
    mins_per_window = 5
    search_range = int(win_scope_mins / mins_per_window / 2)
    for jdx in range(out_arr.size):
        search_arr = signal_col[max(0, jdx - search_range):min(out_arr.size, jdx + search_range)]
        out_arr[jdx] = np.percentile(search_arr, percentile)
    return out_arr


def subsample_blocked(val_arr, meas_arr=None, ss_ratio=None, block_length=None, normal_value=None):
    """ Subsample blocked with ratio and block length"""
    val_arr_out = np.copy(val_arr)
    meas_arr_out = np.copy(meas_arr)
    meas_idxs = []
    n_meas = 0

    for idx in range(meas_arr.size):
        if meas_arr[idx] > n_meas:
            meas_idxs.append(idx)
            n_meas += 1

    if len(meas_idxs) == 0:
        return (val_arr_out, meas_arr_out)

    meas_select = int((1 - ss_ratio) * len(meas_idxs))
    begin_select = meas_select // block_length
    feas_begins = [meas_idxs[idx] for idx in np.arange(0, len(meas_idxs), block_length)]
    sel_meas_begins = sorted(random.sample(feas_begins, begin_select))
    sel_meas_delete = []
    for begin in sel_meas_begins:
        for add_idx in range(block_length):
            sel_meas_delete.append(begin + add_idx)

    # Rewrite the measuremnent array with deleted indices
    for midx, meas_idx in enumerate(meas_idxs):
        prev_cnt = 0 if meas_idx == 0 else meas_arr_out[meas_idx - 1]
        revised_cnt = prev_cnt if meas_idx in sel_meas_delete else prev_cnt + 1
        if midx < len(meas_idxs) - 1:
            for rewrite_idx in range(meas_idx, meas_idxs[midx + 1]):
                meas_arr_out[rewrite_idx] = revised_cnt
        else:
            for rewrite_idx in range(meas_idx, len(meas_arr_out)):
                meas_arr_out[rewrite_idx] = revised_cnt

    # Rewrite the value array with deleted indices, with assuming forward filling
    for midx, meas_idx in enumerate(meas_idxs):
        prev_val = normal_value if meas_idx == 0 else val_arr_out[meas_idx - 1]
        cur_val = val_arr_out[meas_idx]
        revised_value = prev_val if meas_idx in sel_meas_delete else cur_val
        if midx < len(meas_idxs) - 1:
            for rewrite_idx in range(meas_idx, meas_idxs[midx + 1]):
                val_arr_out[rewrite_idx] = revised_value
        else:
            for rewrite_idx in range(meas_idx, len(meas_arr_out)):
                val_arr_out[rewrite_idx] = revised_value

    return (val_arr_out, meas_arr_out)


def subsample_individual(val_arr, meas_arr=None, ss_ratio=None, normal_value=None):
    """ Subsample individual measurements completely randomly with random choice"""
    val_arr_out = np.copy(val_arr)
    meas_arr_out = np.copy(meas_arr)
    meas_idxs = []
    n_meas = 0

    for idx in range(meas_arr.size):
        if meas_arr[idx] > n_meas:
            meas_idxs.append(idx)
            n_meas += 1

    if len(meas_idxs) == 0:
        return (val_arr_out, meas_arr_out)

    meas_select = int((1 - ss_ratio) * len(meas_idxs))
    sel_meas_delete = sorted(random.sample(meas_idxs, meas_select))

    # Rewrite the measuremnent array with deleted indices
    for midx, meas_idx in enumerate(meas_idxs):
        prev_cnt = 0 if meas_idx == 0 else meas_arr_out[meas_idx - 1]
        revised_cnt = prev_cnt if meas_idx in sel_meas_delete else prev_cnt + 1
        if midx < len(meas_idxs) - 1:
            for rewrite_idx in range(meas_idx, meas_idxs[midx + 1]):
                meas_arr_out[rewrite_idx] = revised_cnt
        else:
            for rewrite_idx in range(meas_idx, len(meas_arr_out)):
                meas_arr_out[rewrite_idx] = revised_cnt

    # Rewrite the value array with deleted indices, with assuming forward filling
    for midx, meas_idx in enumerate(meas_idxs):
        prev_val = normal_value if meas_idx == 0 else val_arr_out[meas_idx - 1]
        cur_val = val_arr_out[meas_idx]
        revised_value = prev_val if meas_idx in sel_meas_delete else cur_val
        if midx < len(meas_idxs) - 1:
            for rewrite_idx in range(meas_idx, meas_idxs[midx + 1]):
                val_arr_out[rewrite_idx] = revised_value
        else:
            for rewrite_idx in range(meas_idx, len(meas_arr_out)):
                val_arr_out[rewrite_idx] = revised_value

    return (val_arr_out, meas_arr_out)


def merge_short_vent_gaps(vent_status_arr, short_gap_hours):
    """ Merge short gaps in the ventilation status array"""
    in_gap = False
    gap_length = 0
    # MH Never used
    before_gap_status = np.nan

    for idx in range(len(vent_status_arr)):
        cur_state = vent_status_arr[idx]
        if in_gap and (cur_state == 0.0 or np.isnan(cur_state)):
            gap_length += 5
        elif not in_gap and (cur_state == 0.0 or np.isnan(cur_state)):
            if idx > 0:
                before_gap_status = vent_status_arr[idx - 1]
            in_gap = True
            in_gap_idx = idx
            gap_length = 5
        elif in_gap and cur_state == 1.0:
            in_gap = False
            after_gap_status = cur_state
            if gap_length / 60. <= short_gap_hours:
                vent_status_arr[in_gap_idx:idx] = 1.0

    return vent_status_arr


def kernel_smooth_arr(input_arr, bandwidth=None):
    """ Kernel smooth an input array with a Nadaraya-Watson kernel smoother"""
    output_arr = np.copy(input_arr)
    fin_arr = output_arr[np.isfinite(output_arr)]
    time_axis = 5 * np.arange(len(output_arr))
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


def delete_short_vent_events(vent_status_arr, short_event_hours):
    """ Delete short events in the ventilation status array"""
    in_event = False
    event_length = 0
    for idx in range(len(vent_status_arr)):
        cur_state = vent_status_arr[idx]
        if in_event and cur_state == 1.0:
            event_length += 5
        if not in_event and cur_state == 1.0:
            in_event = True
            event_length = 5
            event_start_idx = idx
        if in_event and (cur_state == 0.0 or np.isnan(cur_state)):
            in_event = False
            if event_length / 60. < short_event_hours:
                vent_status_arr[event_start_idx:idx] = 0.0
    return vent_status_arr


def ellis(x_orig):
    """ ELLIS model converting SpO2 in 100 % units into a PaO2 ABGA
        estimate"""
    x_orig[np.isnan(x_orig)] = 98  # Normal value assumption
    x = x_orig / 100
    x[x == 1] = 0.999
    exp_base = (11700 / ((1 / x) - 1))
    exp_sqrbracket = np.sqrt(pow(50, 3) + (exp_base ** 2))
    exp_first = np.cbrt(exp_base + exp_sqrbracket)
    exp_second = np.cbrt(exp_base - exp_sqrbracket)
    exp_full = exp_first + exp_second
    return exp_full


def correct_left_edge_vent(vent_status_arr, etco2_meas_cnt, etco2_col):
    """ Corrects the left edge of the ventilation status array, to pin-point the exact conditions"""
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
                    etco2_col[idx] > 0.5:
                on_left_edge = False
            else:
                vent_status_arr[idx] = 0.0

    return vent_status_arr


def delete_small_continuous_blocks(event_arr, block_threshold=None):
    """ Given an event array, deletes small contiguous blocks that are sandwiched between two other blocks, one of which
        is longer, they both have the same label. For the moment we delete blocks smaller than 30 minutes. Note this
        requires only a linear pass over the array"""
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


def collect_regression_data(spo2_col, spo2_meas_cnt, pao2_col, pao2_meas_cnt, fio2_est_arr,
                            sao2_col, sao2_meas_cnt, ph_col, ph_meas_cnt):
    """ Collect regression data at time-stamps where we have a real PaO2 measurement, return
        partial training X,y pairs for this patient"""
    X_arr_collect = []
    y_arr_collect = []
    aux_collect = []
    cur_pao2_cnt = 0
    cur_spo2_cnt = 0
    cur_sao2_cnt = 0
    cur_ph_cnt = 0
    pao2_real_meas = []
    spo2_real_meas = []
    sao2_real_meas = []
    ph_real_meas = []

    for jdx in range(spo2_col.size):

        if spo2_meas_cnt[jdx] > cur_spo2_cnt:
            spo2_real_meas.append(jdx)
            cur_spo2_cnt = spo2_meas_cnt[jdx]
        if sao2_meas_cnt[jdx] > cur_sao2_cnt:
            sao2_real_meas.append(jdx)
            cur_sao2_cnt = sao2_meas_cnt[jdx]
        if ph_meas_cnt[jdx] > cur_ph_cnt:
            ph_real_meas.append(jdx)
            cur_ph_cnt = ph_meas_cnt[jdx]

        if pao2_meas_cnt[jdx] > cur_pao2_cnt:
            pao2_real_meas.append(jdx)
            cur_pao2_cnt = pao2_meas_cnt[jdx]

            # Only start fitting the model from the 2nd measurement onwards
            if len(pao2_real_meas) >= 2 and len(spo2_real_meas) >= 2 and len(sao2_real_meas) >= 2 and len(
                    ph_real_meas) >= 2:

                # Dimensions of features
                # 0: Last real SpO2 measurement
                # 1: Last real PaO2 measurement
                # 2: Last real SaO2 measurement
                # 3: Last real pH measurement
                # 4: Time to last real SpO2 measurement
                # 5: Time to last real PaO2 measurement
                # 6: Closest SpO2 to last real PaO2 measurement
                x_vect = np.array([spo2_col[jdx - 1], pao2_col[jdx - 1], sao2_col[jdx - 1], ph_col[jdx - 1],
                                   jdx - spo2_real_meas[-2], jdx - pao2_real_meas[-2], spo2_col[pao2_real_meas[-2]]])
                y_val = pao2_col[jdx]
                aux_val = fio2_est_arr[jdx]

                if np.isnan(x_vect).sum() == 0 and np.isfinite(y_val) and np.isfinite(aux_val):
                    X_arr_collect.append(x_vect)
                    y_arr_collect.append(y_val)
                    aux_collect.append(aux_val)

    if len(X_arr_collect) > 0:
        X_arr = np.vstack(X_arr_collect)
        y_arr = np.array(y_arr_collect)
        aux_arr = np.array(aux_collect)
        assert (np.isnan(X_arr).sum() == 0 and np.isnan(y_arr).sum() == 0)
        return (X_arr, y_arr, aux_arr)
    else:
        return (None, None, None)


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


def suppox_to_fio2(suppox_val):
    """ Conversion of supplemental oxygen to FiO2 estimated value"""
    if suppox_val > 15:
        return 75
    else:
        return SUPPOX_TO_FIO2[suppox_val]


def conservative_state(state1, state2):
    """ Given two states, return the lower one """
    if state1 == state2:
        return state1
    for skey in ["event_0", "event_1", "event_2"]:
        if state1 == skey or state2 == skey:
            return skey
    return "event_3"


def endpoint_gen_benchmark(configs):
    var_map = configs["VAR_IDS"]
    # MH Never used
    raw_var_map = configs["RAW_VAR_IDS"]
    sz_window = configs["length_fw_window"]
    abga_window = configs["length_ABGA_window"]
    # MH Never used
    missing_unm = 0

    # MH Never used
    # Threshold statistics
    stat_counts_ready_and_failure = 0
    stat_counts_ready_and_success = 0
    stat_counts_nready_and_failure = 0
    stat_counts_nready_and_success = 0
    stat_counts_ready_nextube = 0
    stat_counts_nready_nextube = 0

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

    df_batch = pd.read_parquet(os.path.join(imputed_f, "batch_{}.parquet".format(batch_id)))

    logging.info("Loaded imputed data done...")
    cand_raw_batch = glob.glob(os.path.join(merged_f, "part-{}.parquet".format(batch_id)))
    assert (len(cand_raw_batch) == 1)
    pids = list(df_batch.patientid.unique())

    logging.info("Number of patients in batch: {}".format(len(df_batch.patientid.unique())))
    # MH Never used
    first_write = True
    out_fp = os.path.join(out_folder, "batch_{}.parquet".format(batch_id))

    event_count = {"FIO2_AVAILABLE": 0, "SUPPOX_NO_MEAS_12_HOURS_LIMIT": 0, "SUPPOX_MAIN_VAR": 0, "SUPPOX_HIGH_FLOW": 0,
                   "SUPPOX_NO_FILL_STOP": 0}

    # MH Never used
    readiness_ext_count = 0
    not_ready_ext_count = 0
    readiness_and_extubated_cnt = 0
    extubated_cnt = 0
    df_static = pd.read_parquet(configs["general_data_table_path"])
    X_reg_collect = []
    y_reg_collect = []
    aux_reg_collect = []

    out_dfs = []

    for pidx, pid in enumerate(pids):
        df_pid = df_batch[df_batch["patientid"] == pid]

        if df_pid.shape[0] == 0:
            logging.info("WARNING: No input data for PID: {}".format(pid))
            continue

        df_merged_pid = pd.read_parquet(cand_raw_batch[0], filters=[("patientid", "=", pid)])
        df_merged_pid.sort_values(by="datetime", inplace=True)

        suppox_val = {}
        # MH Never used
        suppox_ts = {}

        # Main route of SuppOx
        df_suppox_red_async = df_merged_pid[[var_map["SuppOx"], "datetime"]]
        df_suppox_red_async = df_suppox_red_async.dropna(how="all", thresh=2)
        suppox_async_red_ts = np.array(df_suppox_red_async["datetime"])

        suppox_val["SUPPOX"] = np.array(df_suppox_red_async[var_map["SuppOx"]])

        # Strategy is to create an imputed SuppOx column based on the spec using
        # forward filling heuristics

        # Relevant meta-variables
        fio2_col = np.array(df_pid[var_map["FiO2"]])
        pao2_col = np.array(df_pid[var_map["PaO2"]])
        etco2_col = np.array(df_pid[var_map["etCO2"]])

        # MH Never used
        paco2_col = np.array(df_pid[var_map["PaCO2"]])
        gcs_a_col = np.array(df_pid[var_map["GCS_Antwort"]])
        gcs_m_col = np.array(df_pid[var_map["GCS_Motorik"]])
        gcs_aug_col = np.array(df_pid[var_map["GCS_Augen"]])
        weight_col = np.array(df_pid[var_map["Weight"][0]])

        noreph_col = np.array(df_pid[var_map["Norephenephrine"][0]])
        epineph_col = np.array(df_pid[var_map["Epinephrine"][0]])
        vaso_col = np.array(df_pid[var_map["Vasopressin"][0]])

        milri_col = np.array(df_pid[var_map["Milrinone"][0]])
        dobut_col = np.array(df_pid[var_map["Dobutamine"][0]])
        levosi_col = np.array(df_pid[var_map["Levosimendan"][0]])
        theo_col = np.array(df_pid[var_map["Theophyllin"][0]])

        lactate_col = np.array(df_pid[var_map["Lactate"][0]])
        peep_col = np.array(df_pid[var_map["PEEP"]])

        # MH Never used
        # Heartrate
        hr_col = np.array(df_pid[var_map["HR"]])
        hr_meas_cnt = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["HR"])])

        # MH Never used
        # Temperature
        temp_col = np.array(df_pid[var_map["Temp"]])
        temp_meas_cnt = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["Temp"])])

        # MH Never used
        rrate_col = np.array(df_pid[var_map["RRate"]])

        tv_col = np.array(df_pid[var_map["TV"]])
        map_col = np.array(df_pid[var_map["MAP"][0]])
        airway_col = np.array(df_pid[var_map["Airway"]])

        # Ventilator mode group columns
        vent_mode_col = np.array(df_pid[var_map["vent_mode"]])

        spo2_col = np.array(df_pid[var_map["SpO2"]])

        if configs["presmooth_spo2"]:
            spo2_col = percentile_smooth(spo2_col, configs["spo2_smooth_percentile"],
                                         configs["spo2_smooth_window_size_mins"])
        # MH Never used
        sao2_col = np.array(df_pid[var_map["SaO2"]])
        ph_col = np.array(df_pid[var_map["pH"]])

        fio2_meas_cnt = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["FiO2"])])
        pao2_meas_cnt = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["PaO2"])])
        etco2_meas_cnt = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["etCO2"])])
        peep_meas_cnt = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["PEEP"])])
        hr_meas_cnt = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["HR"])])
        spo2_meas_cnt = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["SpO2"])])

        # MH Never used
        sao2_meas_cnt = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["SaO2"])])
        ph_meas_cnt = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["pH"])])

        abs_dtime_arr = np.array(df_pid["datetime"])
        event_status_arr = np.zeros(shape=(fio2_col.size), dtype="<S10")

        # Status arrays
        pao2_avail_arr = np.zeros(shape=(fio2_col.size))
        fio2_avail_arr = np.zeros(shape=(fio2_col.size))
        fio2_suppox_arr = np.zeros(shape=(fio2_col.size))
        fio2_ambient_arr = np.zeros(shape=(fio2_col.size))
        pao2_sao2_model_arr = np.zeros(shape=(fio2_col.size))
        pao2_full_model_arr = np.zeros(shape=(fio2_col.size))

        ratio_arr = np.zeros(shape=(fio2_col.size))
        sur_ratio_arr = np.zeros(shape=(fio2_col.size))

        pao2_est_arr = np.zeros(shape=(fio2_col.size))
        fio2_est_arr = np.zeros(shape=(fio2_col.size))
        vent_status_arr = np.zeros(shape=(fio2_col.size))
        readiness_ext_arr = np.zeros(shape=(fio2_col.size))
        readiness_ext_arr[:] = np.nan

        # Votes arrays
        vent_votes_arr = np.zeros(shape=(fio2_col.size))
        vent_votes_etco2_arr = np.zeros(shape=(fio2_col.size))
        vent_votes_ventgroup_arr = np.zeros(shape=(fio2_col.size))
        vent_votes_tv_arr = np.zeros(shape=(fio2_col.size))
        vent_votes_airway_arr = np.zeros(shape=(fio2_col.size))

        peep_status_arr = np.zeros(shape=(fio2_col.size))
        peep_threshold_arr = np.zeros(shape=(fio2_col.size))
        hr_status_arr = np.zeros(shape=(fio2_col.size))
        etco2_status_arr = np.zeros(shape=(fio2_col.size))
        event_status_arr.fill("UNKNOWN")

        # Array pointers tracking the current active value of each type
        suppox_async_red_ptr = -1

        # ======================== VENTILATION ================================================================================================

        # Label each point in the 30 minute window with ventilation

        # MH Never used
        in_vent_event = False

        for jdx in range(0, len(ratio_arr)):
            low_vent_idx = max(0, jdx - configs["peep_search_bw"])
            high_vent_idx = min(len(ratio_arr), jdx + configs["peep_search_bw"])
            low_peep_idx = max(0, jdx - configs["peep_search_bw"])
            high_peep_idx = min(len(ratio_arr), jdx + configs["peep_search_bw"])
            low_hr_idx = max(0, jdx - configs["hr_vent_search_bw"])
            high_hr_idx = min(len(ratio_arr), jdx + configs["hr_vent_search_bw"])

            win_etco2 = etco2_col[low_vent_idx:high_vent_idx]
            win_etco2_meas = etco2_meas_cnt[low_vent_idx:high_vent_idx]
            win_peep = peep_col[low_peep_idx:high_peep_idx]
            win_peep_meas = peep_meas_cnt[low_peep_idx:high_peep_idx]
            win_hr_meas = hr_meas_cnt[low_hr_idx:high_hr_idx]

            etco2_meas_win = win_etco2_meas[-1] - win_etco2_meas[0] > 0
            peep_meas_win = win_peep_meas[-1] - win_peep_meas[0] > 0
            hr_meas_win = win_hr_meas[-1] - win_hr_meas[0] > 0
            current_vent_group = vent_mode_col[jdx]
            current_tv = tv_col[jdx]
            current_airway = airway_col[jdx]

            vote_score = 0

            # EtCO2 requirement (still needed)
            if etco2_meas_win and (win_etco2 > 0.5).any():
                vote_score += 2
                vent_votes_etco2_arr[jdx] = 2

            # Ventilation group requirement (still needed)
            if current_vent_group in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
                vote_score += 1
                vent_votes_ventgroup_arr[jdx] += 1
            elif current_vent_group in [1.0]:
                vote_score -= 1
                vent_votes_ventgroup_arr[jdx] -= 1
            elif current_vent_group in [11.0, 12.0, 13.0, 15.0, 17.0]:
                vote_score -= 2
                vent_votes_ventgroup_arr[jdx] -= 2

            # TV presence requirement (still needed)
            if current_tv > 0:
                vote_score += 1
                vent_votes_tv_arr[jdx] = 1

            # Airway requirement (still needed)
            if current_airway in [1, 2]:
                vote_score += 2
                vent_votes_airway_arr[jdx] = 2

            # No airway (still needed)
            if current_airway in [3, 4, 5, 6]:
                vote_score -= 1
                vent_votes_airway_arr[jdx] = -1

            vent_votes_arr[jdx] = vote_score

            if vote_score >= configs["vent_vote_threshold"]:
                # MH Never used
                in_vent_event = True

                vent_status_arr[jdx] = 1
            else:
                # MH Never used
                in_vent_event = False

            if peep_meas_win:
                peep_status_arr[jdx] = 1
            if (win_peep >= configs["peep_threshold"]).any():
                peep_threshold_arr[jdx] = 1
            if etco2_meas_win:
                etco2_status_arr[jdx] = 1
            if hr_meas_win:
                hr_status_arr[jdx] = 1

        if configs["detect_hr_gaps"]:
            vent_status_arr = delete_low_density_hr_gap(vent_status_arr, hr_status_arr, configs=configs)

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
                event_length += 5
            if not in_event and cur_state == 1.0:
                in_event = True
                event_length = 5
                event_start_idx = idx
            if in_event and (np.isnan(cur_state) or cur_state == 0.0):
                in_event = False

                # Short event at beginning of stay shall never be deleted...
                if event_start_idx == 0:
                    delete_event = False
                else:
                    search_hr_idx = event_start_idx - 1
                    while search_hr_idx >= 0:
                        if hr_status_arr[search_hr_idx] == 1.0:
                            hr_gap_length = 5 * (event_start_idx - search_hr_idx)
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

        # ============================== OXYGENATION ENDPOINTS ==================================================================

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
            bw_fio2 = fio2_col[max(0, jdx - configs["sz_fio2_window"]):jdx + 1]
            bw_fio2_meas = fio2_meas_cnt[max(0, jdx - configs["sz_fio2_window"]):jdx + 1]
            bw_etco2_meas = etco2_meas_cnt[max(0, jdx - configs["sz_etco2_window"]):jdx + 1]
            fio2_meas = bw_fio2_meas[-1] - bw_fio2_meas[0] > 0

            # MH Never used
            etco2_meas = bw_etco2_meas[-1] - bw_etco2_meas[0] > 0

            mode_group_est = vent_mode_col[jdx]

            # FiO2 is measured since beginning of stay and EtCO2 was measured, we use FiO2 (indefinite forward filling)
            # if ventilation is active or the current estimate of ventilation mode group is NIV.
            if fio2_meas and (vent_status_arr[jdx] == 1.0 or mode_group_est == 4.0):
                event_count["FIO2_AVAILABLE"] += 1
                fio2_val = bw_fio2[-1] / 100
                fio2_avail_arr[jdx] = 1

            # Use supplemental oxygen or ambient air oxygen
            else:

                # No real measurements up to now, or the last real measurement
                # was more than 8 hours away.
                if suppox_async_red_ptr == -1 or (
                        cur_time - suppox_async_red_ts[suppox_async_red_ptr]) > np.timedelta64(
                    configs["suppox_max_ffill"], 'h'):
                    event_count["SUPPOX_NO_MEAS_12_HOURS_LIMIT"] += 1
                    fio2_val = configs["ambient_fio2"]
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
                        assert (False, "Impossible condition")

            bw_pao2_meas = pao2_meas_cnt[max(0, jdx - configs["sz_pao2_window"]):jdx + 1]
            bw_pao2 = pao2_col[max(0, jdx - configs["sz_pao2_window"]):jdx + 1]
            pao2_meas = bw_pao2_meas[-1] - bw_pao2_meas[0] >= 1

            # PaO2 was just measured, just use the value
            if pao2_meas:
                pao2_estimate = bw_pao2[-1]
                pao2_avail_arr[jdx] = 1

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
                    spo2_val = 98
                    pao2_estimate = ellis(np.array([spo2_val]))[0]

            # Compute the individual components of the Horowitz index
            pao2_est_arr[jdx] = pao2_estimate
            fio2_est_arr[jdx] = fio2_val

        # MH Never used
        pao2_est_arr_orig = np.copy(pao2_est_arr)

        # Smooth individual components of the P/F ratio estimate
        if configs["kernel_smooth_estimate_pao2"]:
            pao2_est_arr = kernel_smooth_arr(pao2_est_arr, bandwidth=configs["smoothing_bandwidth"])

        if configs["kernel_smooth_estimate_fio2"]:
            fio2_est_arr = kernel_smooth_arr(fio2_est_arr, bandwidth=configs["smoothing_bandwidth"])

        # Test2 data-set for surrogate model
        pao2_sur_est = np.copy(pao2_est_arr)
        assert (np.sum(np.isnan(pao2_sur_est)) == 0)

        # Convex combination of the estimate
        if configs["mix_real_estimated_pao2"]:
            pao2_est_arr = mix_real_est_pao2(pao2_col, pao2_meas_cnt, pao2_est_arr,
                                             bandwidth=configs["smoothing_bandwidth"])

        # Compute Horowitz indices (Kernel pipeline / Surrogate model pipeline)
        for jdx in range(len(ratio_arr)):
            ratio_arr[jdx] = pao2_est_arr[jdx] / fio2_est_arr[jdx]

        # Post-smooth Horowitz index
        if configs["post_smooth_pf_ratio"]:
            ratio_arr = kernel_smooth_arr(ratio_arr, bandwidth=configs["post_smoothing_bandwidth"])

        if configs["pao2_version"] == "ellis_basic":
            pf_event_est_arr = np.copy(ratio_arr)
        elif configs["pao2_version"] == "original":
            assert (False)

        # Now label based on the array of estimated Horowitz indices
        for idx in range(0, len(event_status_arr) - configs["offset_back_windows"]):
            est_idx = pf_event_est_arr[idx:min(len(ratio_arr), idx + sz_window)]
            est_vent = vent_status_arr[idx:min(len(ratio_arr), idx + sz_window)]
            est_peep_dense = peep_status_arr[idx:min(len(ratio_arr), idx + sz_window)]
            est_peep_threshold = peep_threshold_arr[idx:min(len(ratio_arr), idx + sz_window)]

            if np.sum((est_idx <= 100) & (
                    (est_vent == 0.0) | (est_vent == 1.0) & (est_peep_dense == 0.0) | (est_vent == 1.0) & (
                    est_peep_dense == 1.0) & (est_peep_threshold == 1.0))) >= 2 / 3 * len(est_idx):
                event_status_arr[idx] = "event_3"
            elif np.sum((est_idx <= 200) & (
                    (est_vent == 0.0) | (est_vent == 1.0) & (est_peep_dense == 0.0) | (est_vent == 1.0) & (
                    est_peep_dense == 1.0) & (est_peep_threshold == 1.0))) >= 2 / 3 * len(est_idx):
                event_status_arr[idx] = "event_2"
            elif np.sum((est_idx <= 300) & (
                    (est_vent == 0.0) | (est_vent == 1.0) & (est_peep_dense == 0.0) | (est_vent == 1.0) & (
                    est_peep_dense == 1.0) & (est_peep_threshold == 1.0))) >= 2 / 3 * len(est_idx):
                event_status_arr[idx] = "event_1"
            elif np.sum(np.isnan(est_idx)) < 2 / 3 * len(est_idx):
                event_status_arr[idx] = "event_0"

        # Re-traverse the array and correct the right edges of events

        # Correct right edges of event 0 (correct level to level 0)
        on_right_edge = False
        in_event = False
        for idx in range(0, len(event_status_arr) - configs["offset_back_windows"]):
            cur_state = event_status_arr[idx].decode()
            if cur_state in ["event_0"] and not in_event:
                in_event = True
            elif in_event and cur_state not in ["event_0"]:
                in_event = False
                on_right_edge = True
            if on_right_edge:
                if pf_event_est_arr[idx] < 300:
                    on_right_edge = False
                else:
                    event_status_arr[idx] = "event_0"

        # Correct right edges of event 1 (correct to level 1)
        on_right_edge = False
        in_event = False
        for idx in range(0, len(event_status_arr) - configs["offset_back_windows"]):
            cur_state = event_status_arr[idx].decode()
            if cur_state in ["event_1"] and not in_event:
                in_event = True
            elif in_event and cur_state not in ["event_1"]:
                in_event = False
                on_right_edge = True
            if on_right_edge:
                if pf_event_est_arr[idx] < 200 or pf_event_est_arr[idx] >= 300:
                    on_right_edge = False
                else:
                    event_status_arr[idx] = "event_1"

        # Correct right edges of event 2 (correct to level 2)
        on_right_edge = False
        in_event = False
        for idx in range(0, len(event_status_arr) - configs["offset_back_windows"]):
            cur_state = event_status_arr[idx].decode()
            if cur_state in ["event_2"] and not in_event:
                in_event = True
            elif in_event and cur_state not in ["event_2"]:
                in_event = False
                on_right_edge = True
            if on_right_edge:
                if pf_event_est_arr[idx] < 100 or pf_event_est_arr[idx] >= 200:
                    on_right_edge = False
                else:
                    event_status_arr[idx] = "event_2"

        # Correct right edges of event 3 (correct to level 3)
        on_right_edge = False
        in_event = False
        for idx in range(0, len(event_status_arr) - configs["offset_back_windows"]):
            cur_state = event_status_arr[idx].decode()
            if cur_state in ["event_3"] and not in_event:
                in_event = True
            elif in_event and cur_state not in ["event_3"]:
                in_event = False
                on_right_edge = True
            if on_right_edge:
                if pf_event_est_arr[idx] >= 100:
                    on_right_edge = False
                else:
                    event_status_arr[idx] = "event_3"

        circ_status_arr = np.zeros_like(map_col)

        # Computation of the circulatory failure toy version of the endpoint
        for jdx in range(0, len(event_status_arr)):
            map_subarr = map_col[max(0, jdx - 12):min(jdx + 12, len(event_status_arr))]
            lact_subarr = lactate_col[max(0, jdx - 12):min(jdx + 12, len(event_status_arr))]
            milri_subarr = milri_col[max(0, jdx - 12):min(jdx + 12, len(event_status_arr))]
            dobut_subarr = dobut_col[max(0, jdx - 12):min(jdx + 12, len(event_status_arr))]
            levosi_subarr = levosi_col[max(0, jdx - 12):min(jdx + 12, len(event_status_arr))]
            theo_subarr = theo_col[max(0, jdx - 12):min(jdx + 12, len(event_status_arr))]
            noreph_subarr = noreph_col[max(0, jdx - 12):min(jdx + 12, len(event_status_arr))]
            epineph_subarr = epineph_col[max(0, jdx - 12):min(jdx + 12, len(event_status_arr))]
            vaso_subarr = vaso_col[max(0, jdx - 12):min(jdx + 12, len(event_status_arr))]
            map_crit_arr = ((map_subarr < 65) | (milri_subarr > 0) | (dobut_subarr > 0) | (levosi_subarr > 0) | (
                    theo_subarr > 0) | (noreph_subarr > 0) | \
                            (epineph_subarr > 0) | (vaso_subarr > 0))
            lact_crit_arr = (lact_subarr > 2)
            if np.sum(map_crit_arr) >= 2 / 3 * len(map_crit_arr) and np.sum(lact_crit_arr) >= 2 / 3 * len(map_crit_arr):
                circ_status_arr[jdx] = 1.0

        # Traverse the array and delete short gap
        event_status_arr, relabel_arr = delete_small_continuous_blocks(event_status_arr,
                                                                       block_threshold=configs[
                                                                           "pf_event_merge_threshold"])

        time_col = np.array(df_pid["datetime"])
        rel_time_col = np.array(df_pid["rel_datetime"])
        pid_col = np.array(df_pid["patientid"])

        df_out_dict = {}

        df_out_dict["datetime"] = time_col
        df_out_dict["rel_datetime"] = rel_time_col
        df_out_dict["patientid"] = pid_col
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
        out_dfs.append(df_out)

    all_df = pd.concat(out_dfs, axis=0)
    all_df.to_parquet(out_fp)
