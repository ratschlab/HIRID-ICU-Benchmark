from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from icu_benchmarks.common import lookups
from icu_benchmarks.common.constants import PID, VARID, VALUE, DATETIME
from icu_benchmarks.preprocessing import merge

TEST_ROOT = Path(__file__).parent.parent

PREPROCESSING_RES = TEST_ROOT.parent / 'preprocessing' / 'resources'


@pytest.fixture()
def varref():
    varref, _ = lookups.read_reference_table(PREPROCESSING_RES / 'varref.tsv')
    return varref

def test_aggregate_cols(varref):
        
    temp_mid = 2
    temp_ids = [400, 410, 7100]
    temp_val = [[37,np.nan, np.nan, 36],
                [np.nan, 38, np.nan, np.nan],
                [np.nan,np.nan,37.5, 38]]
    exp_val = [37, 38, 37.5, 37]
    df = pd.DataFrame({f"v{vid}": temp_val[j] 
                       for j,vid in enumerate(temp_ids)})
    
    df_res = merge.aggregate_cols(df, varref)
    assert list(df_res[f"vm{temp_mid}"]) == exp_val
    

def test_drop_out_of_range_values(varref):
    length = 3

    heart_rate_id = 200
    invalid_val = -10
    df = pd.DataFrame({
        PID: [1234] * length,
        VARID: [heart_rate_id] * length,
        VALUE: [invalid_val, 68, 300]
    })

    df_res = merge.drop_out_of_range_values(df, varref)
    assert df_res.shape == (2, 3)
    assert all(df_res['value'] != invalid_val)  # invalid value should be gone


def _get_df_with_duplicates(varid, values):
    time = pd.Timestamp('2010-01-01 12:00:00')
    length = len(values)

    df = pd.DataFrame({
        DATETIME: [time] * length,
        VARID: [varid] * length,
        VALUE: values
    })

    return df


@pytest.mark.parametrize("values,expected_values", (
        ([], []),
        ([0.5, 0.5, 0.5], [0.5]),
        ([1.0, 1.1, 1.1, 1.0], [1.05]),
))
def test_drop_duplicates_non_pharma(values, expected_values):
    varid = 23
    stddev_dict = {varid: 20.0}

    df = _get_df_with_duplicates(varid, values)

    df_ret = merge.drop_duplicates_non_pharma(df, stddev_dict)

    assert len(df_ret) == len(expected_values)
    assert list(df_ret[VALUE]) == expected_values

