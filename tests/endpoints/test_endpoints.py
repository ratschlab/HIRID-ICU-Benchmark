from pathlib import Path

import pytest
import numpy as np

from icu_benchmarks.common.constants import STEPS_PER_HOUR
from icu_benchmarks.endpoints import endpoint_benchmark

TEST_ROOT = Path(__file__).parent.parent

PREPROCESSING_RES = TEST_ROOT.parent / 'preprocessing' / 'resources'

#STEPS_PER_HOUR = 12
MINS_PER_STEP = 60 // STEPS_PER_HOUR


#@pytest.mark.parametrize("short_gap_hours", (1))

def test_merge_short_vent_gaps():
    vent_status_arr = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
    short_gap_hours = 1.0

    status = endpoint_benchmark.merge_short_vent_gaps(vent_status_arr, short_gap_hours)

    assert np.all(status == np.array([1., 1., 1., 0., 0.]))
