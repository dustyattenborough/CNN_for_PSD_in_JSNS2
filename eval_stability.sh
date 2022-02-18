#!/bin/bash
# for i in 1614 1598 1621 1635 1679 1680 1687 1693 1694 1695 1711 1719 1770 1784 1796 1799 1797 1798
for i in 1614
do
      python eval_stability.py \
    --config config_total.yaml \
    --batch 32 \
    --mdinput /users/yewzzang/work/JSNS2_22y/result/run1221_r01592_R14_Z10_noshif_nolog_0 \
    -o eval_${i}_model_r01592_R14_Z10_nolog \
    --device 2 \
    --runnum ${i} --rho 1.4 --vtz 1.0 --odd 0 \
    

    mv "/users/yewzzang/work/JSNS2_22y/result/eval_${i}_model_r01592_R14_Z10_nolog/sampleInfo.csv" "/users/yewzzang/work/JSNS2_22y/result/eval_${i}_model_r01592_R14_Z10_nolog/sampleInfo_eval.csv"

done








