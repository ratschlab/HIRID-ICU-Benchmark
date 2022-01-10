source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/transformer.gin \
                             -l logs/Data_Resolution/transformer/ \
                             -t Dynamic_RespFailure_12Hours\
                             -o True \
                             -lr 1e-4\
                             -bs 8\
                             --hidden 64 \
                             --do 0.0 \
                             --do_att 0.3 \
                             --depth 2 \
                             --heads 1 \
			                 -r 1 2 6 12 \
			                 -rl 12 \
			                 -sd 1111 2222 3333 4444 5555



