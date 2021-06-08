source activate icu-benchmark

python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/LogisticRegression.gin \
                             -l logs/random_search/24h_multiclass/LogisticRegression/run \
                             -t Phenotyping_APACHEGroup \
                             --maxlen 288 \
                             -rs True\
                             -sd 1111 2222 3333 \
                             --loss-weight balanced None \
                             --c_parameter 0.001 0.01 0.1 1 10\
			                 --penalty 'l1' 'l2' 'none'
