source activate icu-benchmark
python -m icu_benchmarks.run train \
                             -c configs/hirid/Classification/TCN.gin \
                             -l logs/hirid/ablation/Horizon/TCN/ \
                             -t Dynamic_RespFailure_12Hours\
                             -o True \
                             --hidden 256 \
                             -lr 3e-4\
                             --do 0.0 \
                             --kernel 2 \
                             --reproducible False \
                             --horizon 12 36 72 144 288 576 1152 2016 \
                             -sd 1111 2222 3333 4444 5555 \

