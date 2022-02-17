#!/bin/bash
for j in 1614 1598 1621 1635 1679 1680 1687 1693 1694 1695 1711 1719 1770 1784 1796 1799 1797 1798
do
    cd /store/hep/JSNS2/PulseShapeDiscr/20220202_1/r00${j}/ME

    for i in *.root;
    do
        echo ${j}
        echo ${i:21:4}
   
    


##### comb ME
    python /users/yewzzang/work/JSNS2_22y/combinedWFToHDF.py -i /store/hep/JSNS2/PulseShapeDiscr/20220202_1/r00${j}/ME/comb.debug.r00${j}.f0${i:21:4}.root -o /store/hep/users/yewzzang/JSNS2/com_data/r00${j}/ME_h5/combined.debug.r00${j}.f0${i:21:4}_ME.h5 --outputpath /store/hep/users/yewzzang/JSNS2/com_data/r00${j}/ME_h5/

    
    
 
    ##### comb FN
    python /users/yewzzang/work/JSNS2_22y/combinedWFToHDF.py -i /store/hep/JSNS2/PulseShapeDiscr/20220202_1/r00${j}/FN/comb.debug.r00${j}.f0${i:21:4}.root -o /store/hep/users/yewzzang/JSNS2/com_data/r00${j}/FN_h5/combined.debug.r00${j}.f0${i:21:4}_FN.h5 --outputpath /store/hep/users/yewzzang/JSNS2/com_data/r00${j}/FN_h5/

    done
done








