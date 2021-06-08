source activate icu-benchmark

python -m icu_benchmarks.run train \
                             -c configs/hirid/Regression/transformer.gin \
                             -l logs/random_search/dynamic_regression/transformer/run \
                             -t Remaining_LOS_Reg Dynamic_UrineOutput_2Hours_Reg \
                             -rs True\
                             -lr  3e-4 1e-4 3e-5 1e-5\
                             -sd 1111 2222 3333 \
                             --hidden 32 64 128 \
                             --do 0.0 0.1 0.2 0.3 0.4 \
                             --do_att 0.0 0.1 0.2 0.3 0.4 \
                             --depth 1 \
                             --heads 1 2 4 8 \

