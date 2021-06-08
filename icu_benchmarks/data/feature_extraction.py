from icu_benchmarks.common import constants
from icu_benchmarks.data.preprocess import get_var_types


def extract_feature_df(df, df_var_ref):
    cat_values, binary_values, _, _ = get_var_types(df.columns, df_var_ref)
    to_exclude = cat_values + binary_values + ['admissiontime', 'sex', 'age', 'height']
    df_feat = feat_extraction(df, to_exclude)
    return df_feat


def feat_extraction(df, exclude=('admissiontime', 'sex', 'age', 'height')):
    cols = df.columns
    assert constants.DATETIME in cols
    assert constants.PID in cols

    def extract_feat(patient_sample):
        cols = [c for c in patient_sample.columns if c not in exclude + [constants.PID, constants.DATETIME]]
        max_cols = ['max_' + c for c in cols]
        min_cols = ['min_' + c for c in cols]
        mean_cols = ['mean_' + c for c in cols]
        meas_cols = ['n_meas_' + c for c in cols]
        patient_sample[max_cols] = patient_sample[cols].cummax().ffill().values
        patient_sample[min_cols] = patient_sample[cols].cummin().ffill().values
        patient_sample[meas_cols] = (patient_sample[cols] / patient_sample[cols]).cumsum().ffill().values
        patient_sample[mean_cols] = patient_sample[cols].cumsum().ffill().values / patient_sample[meas_cols].values
        return patient_sample[[constants.PID, constants.DATETIME] + min_cols + max_cols + mean_cols + meas_cols]

    df = df.groupby(constants.PID).apply(lambda x: extract_feat(x))
    df = df.reset_index(level=0, drop=True)
    return df
