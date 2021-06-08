source activate icu-benchmark

python -m icu_benchmarks.run train \
                             -c configs/hirid/Regression/LGBM.gin \
                             -l logs/hirid/random_search_larger_dynamic_regression/LGBM/run \
                             -t Remaining_LOS_Reg Dynamic_UrineOutput_2Hours_Reg \
                             -rs True\
                             -sd 1111 2222 3333 \
                             --depth 3 4 5 6 7 \
                             --subsample-feat 0.33 0.66 1.00 \
                             --subsample-data 0.33 0.66 1.00
