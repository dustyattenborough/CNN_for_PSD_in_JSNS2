#!/bin/bash


for i in 1614 1598 1621 1635 1679 1680 1687 1693 1694 1695 1711 1719 1770 1784 1796 1799 1797 1798
do
    for j in 1.4  ### vertex X^2+Y^2
    do
        
        for m in 1.0 ### vertex Z
        do
            echo ${i}
            echo ${j}

            echo ${m}
            ##################### min value 없음
            python select_vertex.py \
            -i /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/FN_h5/ \
            -o /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/FN_cut_Rho_${j}_ZL_${m}/  \
            --vtxRho ${j} --vtxz ${m} --type 0 --min 1 --minvalue 400 --dvertex 1 --cut 1

            echo "FN clear"

            python select_vertex.py \
            -i /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/ME_h5/ \
            -o /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/ME_cut_Rho_${j}_ZL_${m}/ \
            --vtxRho ${j} --vtxz ${m} --type 1 --min 1 --minvalue 400 --dvertex 1 --cut 1

            echo "ME clear"

            python data_subrun_divide.py \
            -i /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/FN_cut_Rho_${j}_ZL_${m}/ \
            --evenoutput /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/FN_cut_even_Rho_${j}_ZL_${m}/ \
            --oddoutput /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/FN_cut_odd_Rho_${j}_ZL_${m}/

            echo "FN div clear"

            python data_subrun_divide.py \
            -i /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/ME_cut_Rho_${j}_ZL_${m}/  \
            --evenoutput /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/ME_cut_even_Rho_${j}_ZL_${m}/ \
            --oddoutput /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/ME_cut_odd_Rho_${j}_ZL_${m}/

            echo "ME div clear"
    
        done
    done
done







