#!/bin/bash
for i in 1563


do
    python train_PSD_1025.py \
    --config config_odd_trn.yaml \
    --epoch 100 --batch 32 \
    -o date20220427_train_1563_noDIN \
    --device 0 --kernel_size 11 --model 1DCNN3FC2 \
    --runnum 1563 --rho 1.4 --vtz 1.0 --odd 2 \
    --minvalue 400 --dvertex 1 --seed 12345

    mv "/users/yewzzang/work/JSNS2_22y/result/date20220427_train_1563_noDIN/sampleInfo.csv" "/users/yewzzang/work/JSNS2_22y/result/date20220427_train_1563_noDIN/sampleInfo_train.csv"
    python eval_PSD_1025.py \
    --config config_even_eval.yaml \
    --batch 32 \
    -o date20220427_train_1563_noDIN \
    --device 0 \
    --runnum ${i} --rho 1.4 --vtz 1.0 --odd 1 --seed 12345 \


    mv "/users/yewzzang/work/JSNS2_22y/result/date20220427_train_1563_noDIN/sampleInfo.csv" "/users/yewzzang/work/JSNS2_22y/result/date20220427_train_1563_noDIN/sampleInfo_eval.csv"

done




