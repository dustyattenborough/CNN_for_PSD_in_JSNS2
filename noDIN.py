import h5py
import numpy as np
file_name_odd = '/store/hep/users/yewzzang/JSNS2/com_data/r001563/FN_cut_odd_Rho_1.4_ZL_1.0_noDIN/'
file_name_even = '/store/hep/users/yewzzang/JSNS2/com_data/r001563/FN_cut_even_Rho_1.4_ZL_1.0_noDIN/'

i = 47
trigg = 9700


if (i%2 ==1):

    file_name = file_name_odd
else:
    file_name = file_name_even

file_numb = 'combined.debug.r001563.f'+str(f'{i:05}')+'_FN.h5'

file = h5py.File(file_name+file_numb, 'r')





out_run = np.delete(np.array(file['events']['run']),np.where(np.array(file['events']['trigID'])==trigg))
out_subrun = np.delete(np.array(file['events']['subrun']),np.where(np.array(file['events']['trigID'])==trigg))

out_waveform = np.delete(np.array(file['events']['waveform']),np.where(np.array(file['events']['trigID'])==trigg),axis=0)
out_dT = np.delete(np.array(file['events']['dT']),np.where(np.array(file['events']['trigID'])==trigg))
out_dVertex = np.delete(np.array(file['events']['dVertex']),np.where(np.array(file['events']['trigID'])==trigg))
out_nVertex = np.delete(np.array(file['events']['nVertex']),np.where(np.array(file['events']['trigID'])==trigg))
out_vertexX = np.delete(np.array(file['events']['vertexX']),np.where(np.array(file['events']['trigID'])==trigg))
out_vertexY = np.delete(np.array(file['events']['vertexY']),np.where(np.array(file['events']['trigID'])==trigg))
out_vertexZ = np.delete(np.array(file['events']['vertexZ']),np.where(np.array(file['events']['trigID'])==trigg))

out_pCharge = np.delete(np.array(file['events']['pCharge']),np.where(np.array(file['events']['trigID'])==trigg),axis=0)
out_minvalue = np.delete(np.array(file['events']['minvalue']),np.where(np.array(file['events']['trigID'])==trigg))
out_trigID = np.delete(np.array(file['events']['trigID']),np.where(np.array(file['events']['trigID'])==trigg))






file.close()

kwargs = {'dtype':'f4', 'compression':'lzf'}
with h5py.File(file_name+file_numb, 'w', libver='latest', swmr=True) as fout:

#     with h5py.File(output_name, 'w', libver='latest', swmr=True) as fout:
    m = fout.create_group('info')
    m.create_dataset('shape', data=[96, 248], dtype='i4')

    g = fout.create_group('events')
    g.create_dataset('run', data=out_run, dtype='i4')
    g.create_dataset('subrun', data=out_subrun, dtype='i4')
    g.create_dataset('trigID', data=out_trigID, dtype='i4')
    g.create_dataset('waveform', data=out_waveform, chunks=(1, 96, 248), **kwargs)
    g.create_dataset('dT', data=out_dT, dtype='f4')
    g.create_dataset('dVertex', data=out_dVertex, dtype='f4')

    g.create_dataset('nVertex', data=out_nVertex, dtype='i4')
    g.create_dataset('vertexX', data=out_vertexX, **kwargs)
    g.create_dataset('vertexY', data=out_vertexY, **kwargs)
    g.create_dataset('vertexZ', data=out_vertexZ, **kwargs)
    g.create_dataset('minvalue', data=out_minvalue, **kwargs)
    g.create_dataset('pCharge', data=out_pCharge, **kwargs)






file = h5py.File(file_name+file_numb, 'r')


file['events']['run']
