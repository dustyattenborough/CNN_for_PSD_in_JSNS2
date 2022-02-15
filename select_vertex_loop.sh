#!/bin/bash


for i in 1825 1834 1846 1563 1592
do
    for j in 1.4  ### vertex X^2+Y^2
    do
        
        for m in 1.0 ### vertex Z
        do
            echo ${i}
            echo ${j}

            echo ${m}
            ##################### min value 없음
            python select_vertex_involve_raw.py \
            -i /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/FN_h5/ \
            -o /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/FN_cut_Rho_${j}_ZL_${m}/  \
            --vtxRho ${j} --vtxz ${m}

            echo "FN clear"

            python select_vertex_involve_raw.py \
            -i /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/ME_h5/ \
            -o /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/ME_cut_Rho_${j}_ZL_${m}/ \
            --vtxRho ${j} --vtxz ${m}

            echo "ME clear"

            python data_subrun_divide.py \
            -i /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/FN_cut_Rho_${j}_ZL_${m}/ \
            --evenoutput /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/FN_even_Rho_${j}_ZL_${m}/ \
            --oddoutput /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/FN_odd_Rho_${j}_ZL_${m}/

            echo "FN div clear"

            python data_subrun_divide.py \
            -i /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/ME_cut_Rho_${j}_ZL_${m}/  \
            --evenoutput /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/ME_even_Rho_${j}_ZL_${m}/ \
            --oddoutput /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/ME_odd_Rho_${j}_ZL_${m}/

            echo "ME div clear"
    
        done
    done
done







