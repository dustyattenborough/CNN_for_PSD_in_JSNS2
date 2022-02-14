#!/usr/bin/env python


import numpy as np
import h5py
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import shutil


import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, required=True, help='Path to input directory')
parser.add_argument('--evenoutput', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--oddoutput', action='store', type=str, required=True, help='Path to output directory')
args = parser.parse_args()


list_name = os.listdir(args.input)


    
numberofdata = 0     
for i in range(len(list_name)):

    file_name = args.input + list_name[i]
    file = h5py.File(file_name, 'r')
    
    
    numberofdata_ori = len(np.array(file['events']['dT']))
    numberofdata = numberofdata + numberofdata_ori
    if (np.array(file['events']['subrun'])[0] % 2) == 1:
        
        if not os.path.exists(args.oddoutput): os.makedirs(args.oddoutput)
            
        output_file = args.oddoutput + list_name[i]
        out_run = np.array(file['events']['run'])
        out_subrun = np.array(file['events']['subrun'])
        out_trigID = np.array(file['events']['trigID'])
        out_waveform = np.array(file['events']['waveform'])
        out_dT = np.array(file['events']['dT'])
        out_dVertex = np.array(file['events']['dVertex'])
        out_nVertex = np.array(file['events']['nVertex'])
        out_vertexX = np.array(file['events']['vertexX'])
        out_vertexY = np.array(file['events']['vertexY'])
        out_vertexZ = np.array(file['events']['vertexZ'])
        
        out_pCharge = np.array(file['events']['pCharge'])
        out_minvalue = np.array(file['events']['minvalue'])

        kwargs = {'dtype':'f4', 'compression':'lzf'}
        with h5py.File(output_file, 'w', libver='latest', swmr=True) as fout:

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
    else:
        if not os.path.exists(args.evenoutput): os.makedirs(args.evenoutput)
            
        output_file = args.evenoutput + list_name[i]
        out_run = np.array(file['events']['run'])
        out_subrun = np.array(file['events']['subrun'])
        out_trigID = np.array(file['events']['trigID'])
        out_waveform = np.array(file['events']['waveform'])
        out_dT = np.array(file['events']['dT'])
        out_dVertex = np.array(file['events']['dVertex'])
        out_nVertex = np.array(file['events']['nVertex'])
        out_vertexX = np.array(file['events']['vertexX'])
        out_vertexY = np.array(file['events']['vertexY'])
        out_vertexZ = np.array(file['events']['vertexZ'])
        
        out_pCharge = np.array(file['events']['pCharge'])
        out_minvalue = np.array(file['events']['minvalue'])

        kwargs = {'dtype':'f4', 'compression':'lzf'}
        with h5py.File(output_file, 'w', libver='latest', swmr=True) as fout:

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


