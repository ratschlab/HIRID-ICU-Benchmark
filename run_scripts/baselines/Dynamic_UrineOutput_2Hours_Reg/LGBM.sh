source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Regression/LGBM.gin \
                             -l logs/benchmark_exp/LGBM/ \
                             -t Dynamic_UrineOutput_2Hours_Reg\
                             -o True \
                             --depth 4 \
                             --subsample-feat 1.0 \
                             --subsample-data 1.00 \
                             -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000
