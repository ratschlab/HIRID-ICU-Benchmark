import gc
import glob
import logging
import os
import os.path
import pickle
import sys
import argparse

import gin
import matplotlib
import numpy as np
import pandas as pd

from icu_benchmarks.common.constants import PID
import icu_benchmarks.imputation.forward_filling as endpoint_ff

matplotlib.use("pdf")


def value_empty(size, default_val, dtype=None):
    """ Returns a vector filled with elements of a specific value"""

    if dtype is not None:
        tmp_arr = np.empty(size, dtype=dtype)
    else:
        tmp_arr = np.empty(size)

    tmp_arr[:] = default_val
    return tmp_arr


def empty_nan(sz):
    """ Returns an empty NAN vector of specified size"""
    arr = np.empty(sz)
    arr[:] = np.nan
    return arr


def load_pickle(fpath):
    """ Given a file path pointing to a pickle file, yields the object pickled in this file"""
    with open(fpath, 'rb') as fp:
        return pickle.load(fp)


def impute_dynamic_df(patient_df, pid=None, df_static=None, typical_weight_dict=None,
                      median_bmi_dict=None, configs=None, schema_dict=None,
                      interval_median_dict=None, interval_iqr_dict=None):
    """ Transformer method, taking as input a data-frame with irregularly sampled input data. The method
        assumes that the data-frame contains a time-stamp column, and the data-frame is sorted along the first
        axis in non-decreasing order with respect to the timestamp column. Pass the <pid> of the patient stay
        as additional information"""
    static_table = df_static[df_static["patientid"] == pid]
    max_grid_length_secs = configs["max_grid_length_days"] * 24 * 3600

    # No static data, patient is not valid, exclude on-the-fly
    if static_table.shape[0] == 0:
        logging.info("WARNING: No static data in patient table...")
        return None

    # More than one row, select one of the rows arbitrarily
    if static_table.shape[0] > 1:
        logging.info("WARNING: More than one row in static table...")
        static_table = static_table.take([0], axis=0)

    static_height = float(static_table["height"])
    static_gender = str(static_table["sex"].values[0]).strip()
    assert (static_gender in ["F", "M", "U"])

    # If either the endpoints or the features don't exist, log the failure but do nothing, the missing patients can be
    # latter added as a new group to the output H5
    if patient_df.shape[0] == 0:
        logging.info("WARNING: p{} has missing features, skipping output generation...".format(pid))
        return None

    all_keys = list(set(patient_df.columns.values.tolist()).difference(
        set(["datetime", "patientid", "a_temp", "m_pm_1", "m_pm_2"])))

    ts = patient_df["datetime"]
    ts_arr = np.array(ts)
    n_ts = ts_arr.size

    hr = np.array(patient_df["vm1"])
    creat = np.array(patient_df["vm156"])
    urine = np.array(patient_df["vm24"])

    finite_hr = ts_arr[np.isfinite(hr)]
    finite_creat = ts_arr[np.isfinite(creat)]
    finite_urine = ts_arr[np.isfinite(urine)]

    if finite_hr.size == 0:
        logging.info("WARNING: Patient {} has no HR, ignoring patient...".format(pid))
        return None

    # Respiratory / circulatory failure, define grid over ICU stay
    if not configs["extended_grid"]:
        ts_min = ts_arr[np.isfinite(hr)][0]
        ts_max = ts_arr[np.isfinite(hr)][-1]
    else:
        ts_min = ts_arr[np.isfinite(hr)][0]
        ts_max = ts_arr[np.isfinite(hr)][-1]
        if finite_creat.size > 0:
            ts_min = min(ts_min, ts_arr[np.isfinite(creat)][0])
            ts_max = max(ts_max, ts_arr[np.isfinite(creat)][-1])
        if finite_urine.size > 0:
            ts_min = min(ts_min, ts_arr[np.isfinite(urine)][0])
            ts_max = max(ts_max, ts_arr[np.isfinite(urine)][-1])

    max_ts_diff = (ts_max - ts_min) / np.timedelta64(1, 's')
    time_grid = np.arange(0.0, min(max_ts_diff + configs["grid_period"], max_grid_length_secs), configs["grid_period"])
    time_grid_abs = [ts_min + pd.Timedelta(seconds=time_grid[idx]) for idx in range(time_grid.size)]
    imputed_df_dict = {}
    imputed_df_dict[configs["patient_id_key"]] = [int(pid)] * time_grid.size
    imputed_df_dict[configs["rel_datetime_key"]] = time_grid
    imputed_df_dict[configs["abs_datetime_key"]] = time_grid_abs

    # There is nothing to do if the patient has no records, just return...
    if n_ts == 0:
        logging.info("WARNING: p{} has an empty record, skipping output generation...".format(pid))
        return None

    # Initialize the storage for the imputed time grid, NANs for the non-pharma, 0 for pharma.
    for col in all_keys:
        if col[:2] == "pm":
            imputed_df_dict[col] = np.zeros(time_grid.size)
        elif col[:2] == "vm":
            imputed_df_dict[col] = empty_nan(time_grid.size)
        else:
            logging.info("ERROR: Invalid variable type")
            assert (False)

    imputed_df = pd.DataFrame(imputed_df_dict)
    norm_ts = np.array(ts - ts_min) / np.timedelta64(1, 's')
    all_keys.remove("vm131")
    all_keys = ["vm131"] + all_keys

    # Impute all variables independently, with the two relevant cases pharma variable and other variable,
    # distinguishable from the variable prefix. We enforce that weight is the first variable to be imputed, so that
    # its time-gridded information can later be used by other custom formulae imputations that depend on it.
    for var_idx, variable in enumerate(all_keys):
        df_var = patient_df[variable]
        assert (n_ts == df_var.shape[0] == norm_ts.size)
        valid_normal = False

        raw_col = np.array(df_var)
        assert (raw_col.size == norm_ts.size)

        # Set non-observed value to NAN (special mode for the Kidney project)
        if configs["impute_normal_value_as_nan"]:
            global_impute_val = np.nan

        observ_idx = np.isfinite(raw_col)
        observ_ts = norm_ts[observ_idx]
        observ_val = raw_col[observ_idx]

        # No values have been observed for this variable, it has to be imputed using the normal value.
        if observ_val.size == 0:
            est_vals = value_empty(time_grid.size, global_impute_val)
            imputed_df[variable] = est_vals
            imputed_df["{}_IMPUTED_STATUS_CUM_COUNT".format(variable)] = np.zeros(time_grid.size)
            imputed_df["{}_IMPUTED_STATUS_TIME_TO".format(variable)] = value_empty(time_grid.size, -1.0)
            continue

        assert (np.isfinite(observ_val).all())
        assert (np.isfinite(observ_ts).all())

        # Get the correct imputation mode for this variable

        # Indefinite forward filling compatibility with Hugo's pipeline
        if configs["only_indefinite_filling"]:
            imp_mode = "Forward fill indefinite"
        else:
            imp_mode = schema_dict[(variable, "Impute semantics")]

        assert (imp_mode in ["Forward fill indefinite", "Forward fill limited", "Attribute one grid point",
                             "Forward fill manual"])

        if imp_mode == "Forward fill indefinite":
            fill_interval_secs = np.inf
        else:
            assert False

        est_vals, cum_count_ts, time_to_last_ms = endpoint_ff.impute_forward_fill_simple(observ_ts, observ_val,
                                                                                         time_grid,
                                                                                         global_impute_val,
                                                                                         configs["grid_period"],
                                                                                         var_type=None,
                                                                                         fill_interval_secs=fill_interval_secs,
                                                                                         variable_id=variable)

        if not configs["impute_normal_value_as_nan"]:
            assert (np.isfinite(est_vals).all())

        imputed_df[variable] = est_vals
        imputed_df["{}_IMPUTED_STATUS_CUM_COUNT".format(variable)] = cum_count_ts
        imputed_df["{}_IMPUTED_STATUS_TIME_TO".format(variable)] = time_to_last_ms

    return imputed_df


def is_df_sorted(df, colname):
    return (np.array(df[colname].diff().dropna(), dtype=np.float64) >= 0).all()


def execute(configs):
    """ Batch wrapper that loops through the patients of one the 50 batches"""
    batch_idx = configs["batch_idx"]

    merged_reduced_base_path = configs["bern_reduced_merged_path"]
    output_reduced_base_path = configs["bern_imputed_reduced_dir"]

    n_skipped_patients = 0
    cand_files = glob.glob(os.path.join(merged_reduced_base_path, "part-{}.parquet".format(batch_idx)))
    assert (len(cand_files) == 1)
    source_fpath = cand_files[0]
    no_patient_output = 0
    df_static = pd.read_parquet(configs["bern_static_info_path"])
    output_dfs = []
    all_patient_df = pd.read_parquet(source_fpath)

    all_pids = all_patient_df[PID].unique()
    logging.info("Number of patient IDs: {}".format(len(all_pids)))

    for pidx, pid in enumerate(all_pids):
        patient_df = all_patient_df[all_patient_df["patientid"] == pid]

        if patient_df.shape[0] == 0:
            n_skipped_patients += 1

        if not is_df_sorted(patient_df, "datetime"):
            patient_df = patient_df.sort_values(by="datetime", kind="mergesort")

        imputed_df = impute_dynamic_df(patient_df, pid=pid, df_static=df_static, typical_weight_dict=None,
                                       median_bmi_dict=None, configs=configs, schema_dict=None,
                                       interval_median_dict=None, interval_iqr_dict=None)

        # No data could be output for this patient...
        if imputed_df is None:
            no_patient_output += 1
            continue

        output_dir = os.path.join(output_reduced_base_path)

        if not configs["debug_mode"]:
            output_dfs.append(imputed_df)

        gc.collect()

        if (pidx + 1) % 100 == 0:
            logging.info("Batch {}: {:.2f} %".format(batch_idx, (pidx + 1) / len(all_pids) * 100))
            logging.info("Number of skipped patients: {}".format(n_skipped_patients))
            logging.info("Number of no patients output: {}".format(no_patient_output))

    # Now dump the dictionary of all data-frames horizontally merged
    if not configs["debug_mode"]:
        all_dfs = pd.concat(output_dfs, axis=0)
        all_dfs.to_parquet(os.path.join(output_dir, "batch_{}.parquet".format(batch_idx)))

    return 0


@gin.configurable
def parse_gin_args_impute(old_configs, gin_configs=None):
    return {**old_configs, **gin_configs}  # merging dicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--batch_idx", type=int, default=None, help="On which batch should imputation be run?")
    parser.add_argument("--split_key", default=None, help="On which split should imputation be run?")
    parser.add_argument("--run_mode", default=None, help="Execution mode")
    parser.add_argument("--gin_config", default="./gin_configs/impute_dynamic_benchmark.gin",
                        help="GIN config file to use")

    args = parser.parse_args()
    configs = vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs = parse_gin_args_impute(configs)

    assert (configs["run_mode"] in ["CLUSTER", "INTERACTIVE"])
    batch_idx = configs["batch_idx"]

    if configs["run_mode"] == "CLUSTER":
        sys.stdout = open(
            os.path.join(configs["log_dir"], "IMPUTE_{}_{}.stdout".format(configs["endpoint"], batch_idx)), 'w')
        sys.stderr = open(
            os.path.join(configs["log_dir"], "IMPUTE_{}_{}.stderr".format(configs["endpoint"], batch_idx)), 'w')

    execute(configs)
