### JSNS2 PSD code
## File information

### python folder

- python/models/allModels.py : involve all models import(?)
- python/models/WF1DCNN3FC2logModel.py : 1D CNN with log
- python/models/WF1DCNN3FC2Model.py : 1D CNN without log
- python/dataset/WFCh2Dataset_each_norm.py : normalize each waveforms
- python/dataset/WFCh2Dataset_max_norm.py : normalize max waveform
- python/dataset/WFCh2Dataset_each_norm_involve_raw.py : same “each_norm” + additional information (dVertex, minvalue, pCahrge, vertex X & Y &Z)

### config file

config_even_eval.yaml : even dataset eval config

config_even_trn.yaml : even dataset train config

config_odd_eval.yaml : odd dataset eval config

config_odd_trn.yaml : odd dataset train config

config_total.yaml : odd + even dataset train & even config (odd, even 나눌필요가 이제 있나?? 이거만 써도 될듯)

### Make file

combinedWFToHDF.py : combined waveform data (.root file) to .h5 file

combinedWFToHDF_nodT.py : combined waveform data (.root file) to .h5 file (nodT feature, sanghoon new data?)

root_to_h5.sh : loop code combinedWFToHDF.py

data_subrun_divide.py : for train divide odd / even subrun and save

select_vertex.py : data selection (i : input path , o : output path, vtxRho : vertex Rho 0.6, 1.0, 1.4….  , vtxz : vertex Z : 0.6, 1.0, 1.4… , type : FN =0 / ME = 1, min : apply min value =1 / no = 0, minvalue : if apply minvalue define value 400… , dvertex : apply dvertex =1 / no = 0 , cut : apply cut condition=1 / no = 0) **input, output, Rho, Z, type정도만 설정하면 됨**

select_vertex_loop.sh : loop code select_vertex.py

### Training & Validation & Evaluation

train_PSD.py : PSD train (**config** : config file, **o** : output path, **a** : use all events condition, odd : 0=training & evaluation odd+even / 1 = training even & evaluation odd / 2=training odd & evaluation even, **runnum** : data runnumber, **rho** : rho, **vtz** : vertex Z, **minvalue** : minvalue, **dvertex** : 1 use / 0 nouse, device : gpu device number, batch : batch size, seed: random seed) runnum, rho,vtz,minvalue,dvertex는 데이터 불러오고 저장할때 보기편하려고 설정한거라 없어져도 됨

eval_PSD.py : PSD eval

eval_stability.py : eval data stability 

eval_stability.sh : loop for eval_stability.py (mdinput : model input ex.1592)

### ipynb script

stability_dataset.ipynb : stability dataset combined result confirm

data_num_check.ipynb : file event num check

evaluation_plots.ipynb : evaluation plots one run (AUC, CNN score, …)

evaluation_nocombined.ipynb : evaluation plots one run (pixel score?)

evaluation_combined.ipynb : evaluation plots combined (multiple runs)

remove_data_dodo_txt.ipynb : remove nonDIN event

model_stability_plots.ipynb : datasets stability comparing