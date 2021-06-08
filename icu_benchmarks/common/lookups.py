import logging
from pathlib import Path

import numpy as np
import pandas as pd

from icu_benchmarks.common.constants import PID


def read_general_table(path: Path):
    logging.info(f"Reading general table from {path}")
    return pd.read_parquet(path, engine="pyarrow").set_index(PID)


def read_var_ref_table(var_ref_path):
    return pd.read_csv(var_ref_path, sep='\t')


def read_reference_table(varref_path):
    """
    Read variableid-metavariableid mapping table for the merge step
    """

    varref = pd.read_csv(varref_path, sep="\t", encoding='cp1252')

    pharmaref = varref[varref["type"] == "pharma"].rename(columns={"variableid": "pharmaid"})
    enum_ref = {'very short': 5, 'short': 1 * 60, '4h': 4 * 60, '6h': 6 * 60, '12h': 12 * 60, '24h': 24 * 60,
                '3d': 3 * 24 * 60}
    pharmaref.loc[:, "pharmaactingperiod_min"] = pharmaref.pharmaactingperiod.apply(
        lambda x: enum_ref[x] if type(x) == str else np.nan)

    varref = varref[varref["type"] != "pharma"].copy()
    varref.drop(varref.index[varref.variableid.isnull()], inplace=True)
    varref.loc[:, "variableid"] = varref.variableid.astype(int)
    varref.set_index("variableid", inplace=True)
    return varref, pharmaref
