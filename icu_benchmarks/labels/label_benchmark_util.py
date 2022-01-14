import numpy as np

STEPS_PER_HOUR = 12


def transition_to_abs(score_arr, target, lhours, rhours):
    """ Transition to an absolute value from a value below the target"""
    out_arr = np.zeros_like(score_arr)
    for jdx in range(out_arr.size):
        if score_arr[jdx] >= target:
            out_arr[jdx] = np.nan
            continue
        low_idx = min(score_arr.size, jdx + 1 + STEPS_PER_HOUR * lhours)
        high_idx = min(score_arr.size, jdx + STEPS_PER_HOUR * rhours)
        future_arr = score_arr[low_idx:high_idx]
        if future_arr.size == 0:
            out_arr[jdx] = np.nan
        elif np.sum(future_arr >= target) > 0:
            out_arr[jdx] = 1.0
    return out_arr


def dynamic_mortality_at_hours(stay_length, mort_status, at_hours=None):
    """ Mortality at a fixed time-point"""
    out_arr = np.zeros(stay_length)
    out_arr[:] = np.nan
    if stay_length >= 24 * STEPS_PER_HOUR:
        out_arr[24 * STEPS_PER_HOUR - 1] = mort_status
    return out_arr


def transition_to_failure(ann_col, lhours=None, rhours=None):
    """ Transition to failure defined on a binary annotation column"""
    out_arr = np.zeros_like(ann_col)
    for jdx in range(len(out_arr)):
        if np.isnan(ann_col[jdx]) or ann_col[jdx] == 1:
            out_arr[jdx] = np.nan
        elif ann_col[jdx] == 0:
            low_idx = min(ann_col.size, jdx + lhours * STEPS_PER_HOUR)
            high_idx = min(ann_col.size, jdx + rhours * STEPS_PER_HOUR)
            fut_arr = ann_col[low_idx:high_idx]
            if (fut_arr == 1.0).any():
                out_arr[jdx] = 1
    return out_arr


def future_urine_output(urine_col, urine_meas_arr, weight_col, rhours=None):
    """ Regression and binary classification problems on urine output in the future"""
    reg_out_arr = np.zeros_like(urine_col)
    binary_out_arr = np.zeros_like(urine_col)

    for jdx in range(urine_col.size):

        # No valid urine measurement anchor in 2 hours, the task is invalid
        end_of_stay_before_2h = jdx + STEPS_PER_HOUR * rhours >= urine_col.size
        urine_diff = urine_meas_arr[jdx + STEPS_PER_HOUR * rhours] - urine_meas_arr[jdx + STEPS_PER_HOUR * rhours - 1]
        no_measurement_in_2h = urine_diff <= 0
        if end_of_stay_before_2h or no_measurement_in_2h:
            binary_out_arr[jdx] = np.nan
            reg_out_arr[jdx] = np.nan
            continue

        current_weight = weight_col[jdx]

        cum_increase = 0
        for kdx in range(1, rhours * STEPS_PER_HOUR):
            if jdx + kdx >= urine_col.size:
                break
            cum_increase += urine_col[jdx + kdx] / STEPS_PER_HOUR
        std_cum_increase = cum_increase / current_weight
        reg_out_arr[jdx] = std_cum_increase / 2

        # More than 0.5 ml/kg
        if std_cum_increase >= 0.5:
            binary_out_arr[jdx] = 1.0

    return reg_out_arr, binary_out_arr


def increase(score_arr, increase_by, lhours, rhours):
    """Increase over a predition horizon"""
    out_arr = np.zeros_like(score_arr)
    for jdx in range(out_arr.size):
        low_idx = min(score_arr.size, jdx + 1 + STEPS_PER_HOUR * lhours)
        high_idx = min(score_arr.size, jdx + STEPS_PER_HOUR * rhours)
        future_arr = score_arr[low_idx:high_idx]
        if future_arr.size == 0:
            out_arr[jdx] = np.nan
            continue
        last_val = future_arr[-1]
        first_val = future_arr[0]
        if last_val - first_val >= increase_by:
            out_arr[jdx] = 1.0
    return out_arr
