#!/usr/bin/env python


import h5py
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, required=True, help='Path to input directory')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--vtxRho', action='store', type=float, default=0.6, help='x^2+y^2')
parser.add_argument('--vtxz', action='store', type=float, default=0.6, help='vertex Z')
parser.add_argument('--type', action='store', type=int, help='FN =0 / ME = 1')
parser.add_argument('--min', action='store', type=int, default = 1, help='no min cut =0 / min cut = 1')
parser.add_argument('--minvalue', action='store', type=int, default = 400, help='min cut')
parser.add_argument('--dvertex', action='store', type=int, default = 1, help='no dvertex cut =0 / dvertex cut = 1')
args = parser.parse_args()


list_name = os.listdir(args.input)

for i in range(len(list_name)):

    file_name = args.input + list_name[i]
    file = h5py.File(file_name,'r')
    
    if not os.path.exists(args.output): os.makedirs(args.output)
        
    output_name = args.output + list_name[i]
   
    vtxX = np.array(file['events']['vertexX'])
    vtxY = np.array(file['events']['vertexY'])
    vtxZ = np.array(file['events']['vertexZ'])
    minvalue = np.array(file['events']['minvalue'])
    pCharge = np.array(file['events']['pCharge'])
    dVertex = np.array(file['events']['dVertex'])
    
    vtxRho = np.hypot(vtxX, vtxY)
    if args.dvertex == 1:
        if args.min == 1:    
            if args.type == 0:
                isFiducial = (vtxRho <args.vtxRho) & (np.abs(vtxZ) < args.vtxz) & (dVertex < 0.6) & (minvalue < args.minvalue)
            else:
                isFiducial = (vtxRho <args.vtxRho) & (np.abs(vtxZ) < args.vtxz) & (minvalue < args.minvalue)
        else:
            if args.type == 0:
                isFiducial = (vtxRho <args.vtxRho) & (np.abs(vtxZ) < args.vtxz) & (dVertex < 0.6)
            else:
                isFiducial = (vtxRho <args.vtxRho) & (np.abs(vtxZ) < args.vtxz)
                
    else:
        ####### type = 0 and type = 1 same / args.dvervex = 0 <--- no dvertex cut
        if args.min == 1:    
            if args.type == 0:
                isFiducial = (vtxRho <args.vtxRho) & (np.abs(vtxZ) < args.vtxz) & (minvalue < args.minvalue)
            else:
                isFiducial = (vtxRho <args.vtxRho) & (np.abs(vtxZ) < args.vtxz) & (minvalue < args.minvalue)
        else:
            if args.type == 0:
                isFiducial = (vtxRho <args.vtxRho) & (np.abs(vtxZ) < args.vtxz)
            else:
                isFiducial = (vtxRho <args.vtxRho) & (np.abs(vtxZ) < args.vtxz)




    out_run = np.array(file['events']['run'])[isFiducial]

    if out_run.shape[0] == 0:
        continue
    
    out_subrun = np.array(file['events']['subrun'])[isFiducial]
    out_trigID = np.array(file['events']['trigID'])[isFiducial]
    out_waveform = np.array(file['events']['waveform'])[isFiducial]
    out_dT = np.array(file['events']['dT'])[isFiducial]
    out_dVertex = np.array(file['events']['dVertex'])[isFiducial]
  
    out_nVertex = np.array(file['events']['nVertex'])[isFiducial]
    
    out_vertexX = np.array(file['events']['vertexX'])[isFiducial]
    out_vertexY = np.array(file['events']['vertexY'])[isFiducial]
    out_vertexZ = np.array(file['events']['vertexZ'])[isFiducial]
    out_pCharge = np.array(file['events']['pCharge'])[isFiducial]
    
    
    
    
    
    
    kwargs = {'dtype':'f4', 'compression':'lzf'}
#     with h5py.File(output_path, 'w', libver='latest', swmr=True) as fout:
    
    with h5py.File(output_name, 'w', libver='latest', swmr=True) as fout:
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
        g.create_dataset('pCharge', data=out_pCharge, **kwargs)






