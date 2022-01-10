source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/TCN.gin \
                             -l logs/Data_Resolution/TCN/ \
                             -t Dynamic_RespFailure_12Hours\
                             -o True \
                             --hidden 64 \
                             -lr 3e-4\
                             --do 0.4 \
                             --kernel 8 \
			                 -r 1 2 6 12 \
			                 --reproducible False \
			                 -rl 12 \
			                 -sd 1111 2222 3333 4444 5555



