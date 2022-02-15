#!/bin/bash

for i in $(seq -w 0 4868)  ## 1563
# for i in $(seq -w 0 6156)  ## 1592
# for i in $(seq -w 0 1338) ## 1825
# for i in $(seq -w 0 0831) ## 1834
# for i in $(seq -w 0 1000) ## 1846
do
    echo ${i}
    
    


##### comb ME
    python combinedWFToHDF_charge_1214.py -i /store/hep/JSNS2/PulseShapeDiscr/20210809_1/v3/r001563/ME/comb.debug.r001563.f0${i}.root -o com_data/r001563_v3_ch/ME_h5/combined.debug.r001563.f0${i}_ME.h5

    
    
    
    ##### comb FN
    python combinedWFToHDF_charge_1214.py -i /store/hep/JSNS2/PulseShapeDiscr/20210809_1/v3/r001563/FN/comb.debug.r001563.f0${i}.root -o com_data/r001563_v3_ch/FN_h5/combined.debug.r001563.f0${i}_FN.h5

done









