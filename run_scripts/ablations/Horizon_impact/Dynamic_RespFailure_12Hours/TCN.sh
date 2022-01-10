source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/TCN.gin \
                             -l logs/hirid/ablation/Horizon/TCN/ \
                             -t Dynamic_RespFailure_12Hours\
                             -o True \
                             --hidden 64 \
                             -lr 3e-4\
                             --do 0.4 \
                             --kernel 8 \
                             --reproducible False \
                             --horizon 12 36 72 144 288 576 1152 2016 \
                             -sd 1111 2222 3333 4444 5555 \

