#!/usr/bin/env python
import sys, os
import argparse
import awkward as ak

parser = argparse.ArgumentParser(description='Convert pulse shape root files to hdf')
parser.add_argument('-i', '--input', action='store', type=str, required=True, help='input waveform file name')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='output file name')
parser.add_argument('-q', '--quiet', action='store_true', default=False, help='supress progress bar')
args = parser.parse_args()

## Determine run number, subRun number from the filename.
## NOTE: the filename scheme should be unchanged: xxx.debug.rNNNNNN.fMMMMM.root
runNumber, subNumber = args.input.split('.')[-3:-1]
if not args.input.endswith('%s.%s.root' % (runNumber, subNumber)):
    print("!!! Run number and subrun number does not agree. Abort!")
    print("    input file=", args.input)
    os.exit(1)
runNumber = runNumber[1:].lstrip('0')
subNumber = subNumber[1:].lstrip('0')
runNumber = 0 if runNumber == '' else int(runNumber)
subNumber = 0 if subNumber == '' else int(subNumber)

## Open the input files and access to the trees
import uproot
import numpy as np
fComb = uproot.open(args.input)

tComb = fComb['comTree']


out_waveform = np.array(tComb['mVcomspAlign'].array())
nEvent, nCh, nT = out_waveform.shape
out_run = np.ones(nEvent)*runNumber
out_subrun = np.ones(nEvent)*subNumber

out_dT = np.array(tComb['Pair_dTime'].array())
out_dVertex = np.array(tComb['Pair_dVertex'].array())
out_trigID = np.array(tComb['TrigID'].array())

out_vertex = np.array(tComb['RecoVertex'].array())

out_nVertex = np.array(ak.num(tComb['RecoVertex'].array()))

out_vertexX = out_vertex[:,0]
out_vertexY = out_vertex[:,1]
out_vertexZ = out_vertex[:,2]

out_pCharge = np.array(tComb['PMTCharge'].array())

out_minvalue = np.array(tComb['RecoMinValue'].array())
if not args.quiet:
    print("waveforms : shape=", out_waveform.shape)
    print("              min=", out_waveform.min(), "max=", out_waveform.max())

    
print("Saving output...", end="")
if args.output.endswith('.h5'):
    import h5py
    kwargs = {'dtype':'f4', 'compression':'lzf'}
    with h5py.File(args.output, 'w', libver='latest', swmr=True) as fout:
        m = fout.create_group('info')
        m.create_dataset('shape', data=[nCh, nT], dtype='i4')

        g = fout.create_group('events')
        g.create_dataset('run', data=out_run, dtype='i4')
        g.create_dataset('subrun', data=out_subrun, dtype='i4')
        g.create_dataset('trigID', data=out_trigID, dtype='i4')
        g.create_dataset('waveform', data=out_waveform, chunks=(1, nCh, nT), **kwargs)
        g.create_dataset('dT', data=out_dT, dtype='f4')
        g.create_dataset('dVertex', data=out_dVertex, dtype='f4')

        g.create_dataset('nVertex', data=out_nVertex, dtype='i4')
        g.create_dataset('vertexX', data=out_vertexX, **kwargs)
        g.create_dataset('vertexY', data=out_vertexY, **kwargs)
        g.create_dataset('vertexZ', data=out_vertexZ, **kwargs)
        g.create_dataset('pCharge', data=out_pCharge, chunks=(1, nCh), **kwargs)
        g.create_dataset('minvalue', data=out_minvalue, dtype='i4')
        
elif args.output.endswith('.root'):
    if int(uproot.__version__.split('.')[0]) >= 4:
        print("\n!!! Writing out to .root in the uproot is not supported yet")
    else:
        with uproot.recreate(args.output, compression=uproot.LZMA(5)) as fout:
            fout['info'] = fout.newtree({"shape", "int32"})
            fout['info'].extend({'shape': [nCh, nT]})

            fout['events'] = fout.newtree({'run':'i4', 'subrun':'i4', 'trigID':'i4',
                                           'waveform':uproot.newbranch('f4', compression=uproot.LZMA(5)),
                                           'dT':'f4', 'dVertex':'f4',
                                           'nVertex':'i4', 'vertexX':'f4', 'vertexY':'f4', 'vertexZ':'f4','pCharge':uproot.newbranch('f4', compression=uproot.LZMA(5)),'minvalue':'i4'})
            fout['events'].extend({'run':out_run, 'subrun':out_subrun, 'trigID':out_trigID,
                                   'waveform':out_waveform,
                                   'dT':out_dT, 'dVertex':out_dVertex,
                                   'nVertex':out_nVertex, 'vertexX':out_vertexX, 'vertexY':out_vertexY, 'vertexZ':out_vertexZ,'pCharge':out_pCharge,'minvalue':out_minvalue})

print("Done.")

