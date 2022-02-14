#!/bin/bash


for i in 1825 1834 1846 1563 1592
do
    for j in 1.4  ### vertex X^2+Y^2
    do
        for k in 400  #### min value
        do
            for l in 1   ### 1 w/ dvertex 0 w/o dvertex
            do
                for m in 1.0 ### vertex Z
                do
                    echo ${i}
                    echo ${j}
                    echo ${k}
                    echo ${l}
                    echo ${m}
                    ##################### min value 없음
                    python select_vertex_char.py \
                    -i /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/FN_h5/ \
                    -o /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/FN_cut_Rho_${j}_ZL_${m}_min_${k}_dv_${l}/  \
                    --type 0 --min 1 --vtxRho ${j} --vtxz ${m} --minvalue ${k} --dvertex ${l}
                    
                    echo "FN clear"
                    
                    python select_vertex_char.py \
                    -i /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/ME_h5/ \
                    -o /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/ME_cut_Rho_${j}_ZL_${m}_min_${k}_dv_${l}/ \
                    --type 1 --min 1 --vtxRho ${j} --vtxz ${m} --minvalue ${k} --dvertex ${l}
                    
                    echo "ME clear"
                    
                    python data_subrun_divide.py \
                    -i /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/FN_cut_Rho_${j}_ZL_${m}_min_${k}_dv_${l}/ \
                    --evenoutput /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/FN_even_Rho_${j}_ZL_${m}_min_${k}_dv_${l}/ \
                    --oddoutput /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/FN_odd_Rho_${j}_ZL_${m}_min_${k}_dv_${l}/
                    
                    echo "FN div clear"
                    
                    python data_subrun_divide.py \
                    -i /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/ME_cut_Rho_${j}_ZL_${m}_min_${k}_dv_${l}/  \
                    --evenoutput /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/ME_even_Rho_${j}_ZL_${m}_min_${k}_dv_${l}/ \
                    --oddoutput /users/yewzzang/work/JSNS2/NuML/PSD/com_data/r00${i}_v3_ch/ME_odd_Rho_${j}_ZL_${m}_min_${k}_dv_${l}/
                    
                    echo "ME div clear"
                
                done
            done
        done
    done
done







