source activate icu-benchmark

python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/TCN.gin \
                             -l logs/random_search/dynamic_circ/TCN/run \
                             -t  Dynamic_CircFailure_12Hours\
                             -rs True\
                             -lr  3e-4 1e-4 3e-5 1e-5\
                             -sd 1111 2222 3333 \
                             --hidden 32 64 128 256 \
                             --do 0.0 0.1 0.2 0.3 0.4 \
                             --kernel 2 4 8 16 32 \

