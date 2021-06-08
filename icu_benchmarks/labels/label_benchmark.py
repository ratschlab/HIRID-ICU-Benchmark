''' Label generation from the benchmark endpoints'''

import gc
import glob
import logging
import os
import os.path
import pickle

import numpy as np
import pandas as pd

import icu_benchmarks.labels.label_benchmark_util as bern_labels
from icu_benchmarks.common.constants import PID


def load_pickle(fpath):
    ''' Given a file path pointing to a pickle file, yields the object pickled in this file'''
    with open(fpath, 'rb') as fp:
        return pickle.load(fp)


def delete_if_exist(path):
    ''' Deletes a path if it exists on the file-system'''
    if os.path.exists(path):
        os.remove(path)


def create_dir_if_not_exist(path, recursive=False):
    ''' Creates a directory if it does not yet exist in the file system'''
    if not os.path.exists(path):
        if recursive:
            os.makedirs(path)
        else:
            os.mkdir(path)


def is_df_sorted(df, colname):
    return (np.array(df[colname].diff().dropna(), dtype=np.float64) >= 0).all()


def convolve_hr(in_arr, hr_status_arr):
    ''' Convolve an array with a HR status arr'''
    out_arr = np.copy(in_arr)
    out_arr[hr_status_arr == 0] = np.nan
    return out_arr


def gen_label(df_pat, df_endpoint, mort_status=None, apache_group=None, pid=None, configs=None):
    ''' Transform input data-frames to new data-frame with labels'''
    abs_time_col = df_pat[configs["abs_datetime_key"]]
    rel_time_col = df_pat[configs["rel_datetime_key"]]
    patient_col = df_pat[configs["patient_id_key"]]

    hr_col = np.array(df_pat["vm1_IMPUTED_STATUS_CUM_COUNT"])
    hr_status_arr = np.zeros_like(hr_col)
    for jdx in range(hr_col.size):
        if jdx in [0, 1]:
            n_hr_count = hr_col[jdx + 2]
        elif jdx == hr_col.size - 1:
            subarr = hr_col[jdx - 3:jdx + 1]
            n_hr_count = subarr[-1] - subarr[0]
        else:
            subarr = hr_col[jdx - 2:jdx + 2]
            n_hr_count = subarr[-1] - subarr[0]

        hr_status_arr[jdx] = 1 if n_hr_count > 0 else 0

    if df_pat.shape[0] == 0 or df_endpoint.shape[0] == 0:
        logging.info("WARNING: Patient {} has no impute data, skipping...".format(pid), flush=True)
        return None

    df_endpoint.set_index(keys="datetime", inplace=True, verify_integrity=True)
    assert ((df_pat.datetime == df_endpoint.index).all())

    output_df_dict = {}
    output_df_dict[configs["abs_datetime_key"]] = abs_time_col
    output_df_dict[configs["rel_datetime_key"]] = rel_time_col
    output_df_dict[configs["patient_id_key"]] = patient_col

    dynamic_mort_arr = bern_labels.dynamic_mortality_at_hours(len(rel_time_col), mort_status, at_hours=24)
    dynamic_mort_arr = convolve_hr(dynamic_mort_arr, hr_status_arr)

    # a) Mortality, predicted after the first 24h   (1x 10%)
    output_df_dict["Mortality_At24Hours"] = dynamic_mort_arr

    circ_failure_col = np.array(df_endpoint.circ_failure_status)
    dynamic_circ_failure = bern_labels.transition_to_failure(circ_failure_col, lhours=0, rhours=12)
    dynamic_circ_failure = convolve_hr(dynamic_circ_failure, hr_status_arr)

    # b) MAP <65mmHg or Catecholamines, and lactate > 2mmol/l  12h (Cont 6%)
    output_df_dict["Dynamic_CircFailure_12Hours"] = dynamic_circ_failure

    # c) ARDS P/F <300 in next 12h   (Cont. 30%) - i would use the published ellis model for the paO2 estimation
    pre_resp_arr = list(df_endpoint.resp_failure_status)
    ann_resp_arr = np.zeros_like(dynamic_circ_failure)
    for jdx in range(ann_resp_arr.size):
        if pre_resp_arr[jdx] in ["event_1", "event_2", "event_3"]:
            ann_resp_arr[jdx] = 1
        elif pre_resp_arr[jdx] in ["UNKNOWN"]:
            ann_resp_arr[jdx] = np.nan

    dynamic_resp_failure = bern_labels.transition_to_failure(ann_resp_arr, lhours=0, rhours=12)
    dynamic_resp_failure = convolve_hr(dynamic_resp_failure, hr_status_arr)

    output_df_dict["Dynamic_RespFailure_12Hours"] = dynamic_resp_failure

    # d) Urin in the next 2h   (Cont. regression) or (Binary below 0.5)
    weight_col = np.array(df_pat.vm131)
    urine_col = np.array(df_pat.vm24)
    urine_meas_arr = np.array(df_pat["vm24_IMPUTED_STATUS_CUM_COUNT"])

    urine_reg_arr, urine_binary_arr = bern_labels.future_urine_output(urine_col, urine_meas_arr, weight_col, rhours=2)
    urine_reg_arr = convolve_hr(urine_reg_arr, hr_status_arr)
    urine_binary_arr = convolve_hr(urine_binary_arr, hr_status_arr)

    output_df_dict["Dynamic_UrineOutput_2Hours_Reg"] = urine_reg_arr
    output_df_dict["Dynamic_UrineOutput_2Hours_Binary"] = urine_binary_arr

    # e) Apache group (multiclass, phenotyping)
    apache_arr = np.zeros_like(urine_reg_arr)
    apache_arr[:] = np.nan
    if apache_arr.size >= 24 * 12:
        apache_arr[24 * 12 - 1] = apache_group
    apache_arr = convolve_hr(apache_arr, hr_status_arr)

    output_df_dict["Phenotyping_APACHEGroup"] = apache_arr

    # f) Remaining lenght of stay (Kont., regression)
    rem_los = np.linspace(apache_arr.size / 12, 0, num=apache_arr.size)
    rem_los = convolve_hr(rem_los, hr_status_arr)

    output_df_dict["Remaining_LOS_Reg"] = rem_los

    output_df = pd.DataFrame(output_df_dict)
    return output_df


def label_gen_benchmark(configs):
    '''Creation of base labels directly defined on the imputed data / endpoints'''
    label_base_dir = configs["label_dir"]
    endpoint_base_dir = configs["endpoint_dir"]
    imputed_base_dir = configs["imputed_dir"]

    apache_ii_map = configs["APACHE_II_map"]
    apache_iv_map = configs["APACHE_IV_map"]

    df_static = pd.read_parquet(configs["general_data_table_path"])
    all_out_dfs = []

    batch_idx = configs["batch_idx"]

    if not configs["debug_mode"]:
        delete_if_exist(os.path.join(label_base_dir, "batch_{}.parquet".format(batch_idx)))

    patient_path = os.path.join(imputed_base_dir, "batch_{}.parquet".format(batch_idx))
    df_all_pats = pd.read_parquet(patient_path)

    all_pids = df_all_pats[PID].unique()

    logging.info("Number of selected PIDs: {}".format(len(all_pids)))

    cand_files = glob.glob(os.path.join(endpoint_base_dir, "batch_{}.parquet".format(batch_idx)))
    assert (len(cand_files) == 1)
    endpoint_path = cand_files[0]

    df_all_endpoints = pd.read_parquet(endpoint_path)

    if configs["verbose"]:
        logging.info("Number of patient IDs: {}".format(len(all_pids)))

    n_skipped_patients = 0
    for pidx, pid in enumerate(all_pids):

        try:
            mort_code = str(df_static[df_static["patientid"] == pid]["discharge_status"].values[0])
            mort_status = mort_code == "dead"
        except ValueError:
            mort_status = False
        except TypeError:
            mort_status = False

        apache_ii_group = float(df_static[df_static["patientid"] == pid]["APACHE II Group"])
        apache_iv_group = float(df_static[df_static["patientid"] == pid]["APACHE IV Group"])

        if np.isfinite(apache_ii_group) and int(apache_ii_group) in apache_ii_map.keys():
            apache_pat_group = apache_ii_map[int(apache_ii_group)]
        elif np.isfinite(apache_iv_group) and int(apache_iv_group) in apache_iv_map.keys():
            apache_pat_group = apache_iv_map[int(apache_iv_group)]
        else:
            apache_pat_group = np.nan

        if not os.path.exists(patient_path):
            logging.info("WARNING: Patient {} does not exists, skipping...".format(pid))
            n_skipped_patients += 1
            continue

        try:
            df_endpoint = df_all_endpoints[df_all_endpoints["patientid"] == pid]
        except:
            logging.info("WARNING: Issue while reading endpoints of patient {}".format(pid))
            n_skipped_patients += 1
            continue

        df_pat = df_all_pats[df_all_pats["patientid"] == pid]

        if df_pat.shape[0] == 0 or df_endpoint.shape[0] == 0:
            if df_pat.shape[0] == 0:
                logging.info("WARNING: Empty endpoints", flush=True)
            else:
                logging.info("WARNING: Empty imputed data in patient {}".format(pid), flush=True)

            n_skipped_patients += 1
            continue

        if not is_df_sorted(df_endpoint, "datetime"):
            df_endpoint = df_endpoint.sort_values(by="datetime", kind="mergesort")

        # PRECOND seems fine

        df_label = gen_label(df_pat, df_endpoint, mort_status=mort_status, apache_group=apache_pat_group, pid=pid,
                             configs=configs)

        if df_label is None:
            logging.info("WARNING: Label could not be created for PID: {}".format(pid))
            n_skipped_patients += 1
            continue

        assert (df_label.shape[0] == df_pat.shape[0])

        if not configs["debug_mode"]:
            all_out_dfs.append(df_label)

        gc.collect()

        if (pidx + 1) % 100 == 0 and configs["verbose"]:
            logging.info("Progress for batch {}: {:.2f} %".format(batch_idx, (pidx + 1) / len(all_pids) * 100))
            logging.info("Number of skipped patients: {}".format(n_skipped_patients))

    if not configs["debug_mode"]:
        combined_df = pd.concat(all_out_dfs, axis=0)
        output_path = os.path.join(label_base_dir, "batch_{}.parquet".format(batch_idx))
        combined_df.to_parquet(output_path)
