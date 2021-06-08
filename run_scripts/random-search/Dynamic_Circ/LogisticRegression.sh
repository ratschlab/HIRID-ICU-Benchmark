source activate icu-benchmark

python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/LogisticRegression.gin \
                             -l logs/random_search/dynamic_circ/LogisticRegression/run \
                             -t  Dynamic_CircFailure_12Hours \
                             -rs True\
                             --loss-weight balanced None \
                             -sd 1111 2222 3333 \
                             --c_parameter 0.001 0.01 0.1 1 10\
			                 --penalty 'l1' 'l2' 'none'
