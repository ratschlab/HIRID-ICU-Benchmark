source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/GRU.gin \
                             -l logs/Data_Resolution/GRU/ \
                             -t Dynamic_RespFailure_12Hours\
                             -o True \
                             -lr 3e-4\
                             --hidden 64 \
                             --do 0.0 \
                             --depth 3 \
			                 -r 1 2 6 12 \
			                 -rl 12 \
			                 -sd 1111 2222 3333 4444 5555 \



