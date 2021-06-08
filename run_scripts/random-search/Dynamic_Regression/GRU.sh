source activate icu-benchmark

python -m icu_benchmarks.run train \
                             -c configs/hirid/Regression/GRU.gin \
                             -l logs/hirid/random_search_larger_dynamic_regression/GRU/run \
                             -t Remaining_LOS_Reg Dynamic_UrineOutput_2Hours_Reg \
                             -rs True\
                             -sd 1111 2222 3333 \
                             --hidden 32 64 128 256 \
                             --do 0.0 0.1 0.2 0.3 0.4 \
                             --depth 1 2 3 \

