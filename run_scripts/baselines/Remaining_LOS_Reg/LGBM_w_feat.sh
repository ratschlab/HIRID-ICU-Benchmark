source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Regression/LGBM_w_feat.gin \
                             -l logs/benchmark_exp/LGBM_w_feat/ \
                             -t Remaining_LOS_Reg\
                             -o True \
                             --depth 7 \
                             --subsample-feat 0.66 \
                             --subsample-data 0.33 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000
