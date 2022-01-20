from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from icu_benchmarks.common import lookups
from icu_benchmarks.common.constants import PID, VARID, VALUE, DATETIME, PHARMA_DATETIME, INFID, PHARMAID, PHARMA_STATUS, PHARMA_VAL
from icu_benchmarks.preprocessing import merge, preprocess_pharma

TEST_ROOT = Path(__file__).parent.parent

PREPROCESSING_RES = TEST_ROOT.parent / 'preprocessing' / 'resources'


@pytest.fixture()
def varref():
    varref, _ = lookups.read_reference_table(PREPROCESSING_RES / 'varref.tsv')
    return varref

def test_drop_duplicates_pharma():

    # check duplicated stop records, one duplicated record has value 0
    dt_start = np.datetime64('2010-01-01 11:50:00')
    dts = dt_start + np.timedelta64(1,'m') * np.arange(0,100,2)
    dts = np.concatenate((dts, dts[-1:]))
    status_val = [524]+[520]*(len(dts)-3)+[776,776]
    pharma_val = np.ones((len(dts),))
    pharma_val[-1:] = 0
    df = pd.DataFrame({PHARMA_DATETIME: dts,
                       INFID: [0]*len(dts),
                       PHARMAID: [1000379]*len(dts),
                       PHARMA_VAL: pharma_val,
                       PHARMA_STATUS: status_val})
    
    df_res = preprocess_pharma.drop_duplicates_pharma(df)
    assert len(df_res) == 50
    assert df_res[df_res[PHARMA_STATUS]==776][PHARMA_VAL].iloc[0]==1

    # check duplicated stop records
    dt_start = np.datetime64('2010-01-01 11:50:00')
    dts = dt_start + np.timedelta64(1,'m') * np.arange(0,100,2)
    dts = np.concatenate((dts, dts[-1:]))
    status_val = [524]+[520]*(len(dts)-3)+[776,776]
    pharma_val = np.ones((len(dts),))
    df = pd.DataFrame({PHARMA_DATETIME: dts,
                       INFID: [0]*len(dts),
                       PHARMAID: [1000379]*len(dts),
                       PHARMA_VAL: pharma_val,
                       PHARMA_STATUS: status_val})
    
    df_res = preprocess_pharma.drop_duplicates_pharma(df)
    assert len(df_res) == 50
    assert df_res[df_res[PHARMA_STATUS]==776][PHARMA_VAL].iloc[0]==1

    # check duplicated records with different status, one status is 776
    dt_start = np.datetime64('2010-01-01 11:50:00')
    dts = dt_start + np.timedelta64(1,'m') * np.arange(0,100,2)
    dts = np.concatenate((dts, dts[-1:]))
    status_val = [524]+[520]*(len(dts)-3)+[520,776]
    pharma_val = np.ones((len(dts),))
    df = pd.DataFrame({PHARMA_DATETIME: dts,
                       INFID: [0]*len(dts),
                       PHARMAID: [1000379]*len(dts),
                       PHARMA_VAL: pharma_val,
                       PHARMA_STATUS: status_val})
    
    df_res = preprocess_pharma.drop_duplicates_pharma(df)
    assert len(df_res) == 50
    assert (df_res[PHARMA_STATUS]==776).sum() == 1

    # check duplicated records with different status, one status is 776
    dt_start = np.datetime64('2010-01-01 11:50:00')
    dts = dt_start + np.timedelta64(1,'m') * np.arange(0,4,2)
    dts = np.concatenate((dts, dts[-1:]))
    status_val = [780] * len(dts)
    pharma_val = np.ones((len(dts),))
    df = pd.DataFrame({PHARMA_DATETIME: dts,
                       INFID: [0]*len(dts),
                       PHARMAID: [1000379]*len(dts),
                       PHARMA_VAL: pharma_val,
                       PHARMA_STATUS: status_val})
    
    df_res = preprocess_pharma.drop_duplicates_pharma(df)
    assert len(df_res) == 3
    assert df_res[INFID].unique().size == 3
    

def test_length_of_stay_filtering():
    admtime = pd.Timestamp('2010-01-01 12:00:00')
    dt = pd.date_range('2010-01-01 11:50:00', periods=20, freq='2T')
    temp_val = np.ones((len(dt), )) * 37

    # heart rate measurement before  admission
    hr_val = np.ones((len(dt), )) * 70
    df = pd.DataFrame({DATETIME: dt,
                       "vm1": hr_val,
                       "vm2": temp_val})
    df_res = merge.length_of_stay_filtering(df, admtime)
    assert len(df_res)==15

    # heart rate measurement after admission and only last 8 steps
    hr_val = np.ones((len(dt), )) * 70
    hr_val[:10] = np.nan
    hr_val[-2:] = np.nan
    df = pd.DataFrame({DATETIME: dt,
                       "vm1": hr_val,
                       "vm2": temp_val})
    df_res = merge.length_of_stay_filtering(df, admtime)
    assert len(df_res)==8

    # no heart rate measurements
    hr_val[:] = np.nan
    df = pd.DataFrame({DATETIME: dt,
                       "vm1": hr_val,
                       "vm2": temp_val})
    df_res = merge.length_of_stay_filtering(df, admtime)
    assert len(df_res)==15

    
    
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

