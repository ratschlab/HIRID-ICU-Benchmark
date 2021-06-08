import numpy as np


def transition_to_abs(score_arr, target, lhours, rhours):
    ''' Transition to an absolute value from a value below the target'''
    out_arr = np.zeros_like(score_arr)
    for jdx in range(out_arr.size):
        if score_arr[jdx] >= target:
            out_arr[jdx] = np.nan
            continue
        future_arr = score_arr[min(score_arr.size, jdx + 1 + 12 * lhours):min(score_arr.size, jdx + 12 * rhours)]
        if future_arr.size == 0:
            out_arr[jdx] = np.nan
        elif np.sum(future_arr >= target) > 0:
            out_arr[jdx] = 1.0
    return out_arr


def dynamic_mortality_at_hours(stay_length, mort_status, at_hours=None):
    ''' Mortality at a fixed time-point'''
    steps_per_hour = 12
    out_arr = np.zeros(stay_length)
    out_arr[:] = np.nan
    if stay_length >= 24 * 12:
        out_arr[24 * 12 - 1] = mort_status
    return out_arr


def transition_to_failure(ann_col, lhours=None, rhours=None):
    ''' Transition to failure defined on a binary annotation column'''
    out_arr = np.zeros_like(ann_col)
    for jdx in range(len(out_arr)):
        if np.isnan(ann_col[jdx]) or ann_col[jdx] == 1:
            out_arr[jdx] = np.nan
        elif ann_col[jdx] == 0:
            fut_arr = ann_col[min(ann_col.size, jdx + lhours * 12):min(ann_col.size, rhours * 12)]
            if (fut_arr == 1.0).any():
                out_arr[jdx] = 1
    return out_arr


def future_urine_output(urine_col, urine_meas_arr, weight_col, rhours=None):
    ''' Regression and binary classification problems on urine output in the future'''
    reg_out_arr = np.zeros_like(urine_col)
    binary_out_arr = np.zeros_like(urine_col)

    for jdx in range(urine_col.size):

        # No valid urine measurement anchor in 2 hours, the task is invalid
        if jdx + 12 * rhours >= urine_col.size or urine_meas_arr[jdx + 12 * rhours] - urine_meas_arr[
            jdx + 12 * rhours - 1] <= 0:
            binary_out_arr[jdx] = np.nan
            reg_out_arr[jdx] = np.nan
            continue

        current_weight = weight_col[jdx]

        cum_increase = 0
        for kdx in range(1, rhours * 12):
            if jdx + kdx >= urine_col.size:
                break
            cum_increase += urine_col[jdx + kdx] / 12
        std_cum_increase = cum_increase / current_weight
        reg_out_arr[jdx] = std_cum_increase / 2

        # More than 0.5 ml/kg
        if std_cum_increase >= 0.5:
            binary_out_arr[jdx] = 1.0

    return (reg_out_arr, binary_out_arr)


def increase(score_arr, increase_by, lhours, rhours):
    ''' Increase over a predition horizon'''
    out_arr = np.zeros_like(score_arr)
    for jdx in range(out_arr.size):
        future_arr = score_arr[min(score_arr.size, jdx + 1 + 12 * lhours):min(score_arr.size, jdx + 12 * rhours)]
        if future_arr.size == 0:
            out_arr[jdx] = np.nan
            continue
        last_val = future_arr[-1]
        first_val = future_arr[0]
        if last_val - first_val >= increase_by:
            out_arr[jdx] = 1.0
    return out_arr
