#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2026  Steven Tavis

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

some code included in this document (PartialPipeline) was released under the Apache 2.0 license
at https://github.com/koaning/tokenwiser
This license can be found in thirdparty_licenses/tokenwiser
"""

import os
import zlib
from collections import defaultdict

import torch
import h5py
import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser
from scipy.stats import binned_statistic
from sklearn.pipeline import Pipeline
from sortedcontainers import SortedList

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype = torch.bfloat16)
    
    def __len__(self):
        return self.X.shape[0]
 
    def __getitem__(self, idx):
        return self.X[idx, :]

def binned_imzML_reader(imzML, 
                        mz_min, 
                        mz_max,
                        ibd = None,
                        bin_width = 0.05,
                        filter_empty_bins = False,
                        drop_uniform_z = True):
    metadata = []
    vectors = []
    if ibd is None:
        data = ImzMLParser(imzML)
    else:
        data = ImzMLParser(imzML, ibd_file = ibd)
    #pyimzml does not natively handle zlib compressed data
    #so we have to detect and decompress manually
    params = data.metadata.pretty()['referenceable_param_groups']
    zlib_i = params['intensityArray']['zlib compression']
    zlib_m = params['mzArray']['zlib compression']
    for idx, (x,y,z) in enumerate(data.coordinates):
        mz_bytes, intensity_bytes = data.get_spectrum_as_string(idx)
        if zlib_i:
            intensity_bytes = zlib.decompress(intensity_bytes)
        if zlib_m:
            mz_bytes = zlib.decompress(mz_bytes)
        intensity = np.frombuffer(intensity_bytes, 
                                  dtype=data.intensityPrecision)
        mz = np.frombuffer(mz_bytes, 
                           dtype=data.mzPrecision)
        vector = binned_statistic(mz,
                                  intensity, 
                                  statistic = 'sum',
                                  bins = int((mz_max - mz_min)/bin_width),
                                  range = (mz_min, mz_max))
        metadata.append(pd.Series({'x':x,
                                   'y':y,
                                   'z':z,
                                   'file':imzML}))
        vectors.append(vector.statistic)
    pixels = pd.DataFrame(metadata)
    vectors = np.array(vectors)
    if filter_empty_bins:
        vector_mask = np.any(vectors > 0, axis = 0)
        vectors = vectors[:,vector_mask]
        mz_cols = vector.bin_edges[:-1][vector_mask]
    else:
        mz_cols = vector.bin_edges[:-1]
    mz_cols += bin_width/2
    mz_cols = [f'mz_{mz}' for mz in mz_cols]
    vectors = pd.DataFrame(vectors, columns = mz_cols)
    pixels = pd.concat([pixels, vectors], axis = 1)
    if drop_uniform_z and len(set(pixels['z'])) == 1:
        del pixels['z']
    return pixels

def targeted_imzML_reader(imzML, 
                          targets,
                          target_names = None,
                          tol = 0.05,
                          reduce = sum,
                          ibd = None,
                          drop_uniform_z = True):
    metadata = []
    vectors = []
    if ibd is None:
        data = ImzMLParser(imzML)
    else:
        data = ImzMLParser(imzML, ibd_file = ibd)
    #pyimzml does not natively handle zlib compressed data
    #so we have to detect and decompress manually
    params = data.metadata.pretty()['referenceable_param_groups']
    zlib_i = params['intensityArray']['zlib compression']
    zlib_m = params['mzArray']['zlib compression']
    for idx, (x,y,z) in enumerate(data.coordinates):
        mz_bytes, intensity_bytes = data.get_spectrum_as_string(idx)
        if zlib_i:
            intensity_bytes = zlib.decompress(intensity_bytes)
        if zlib_m:
            mz_bytes = zlib.decompress(mz_bytes)
        intensity = np.frombuffer(intensity_bytes, 
                                  dtype=data.intensityPrecision)
        mz = np.frombuffer(mz_bytes, 
                           dtype=data.mzPrecision)
        peaks = SortedList(zip(mz, intensity))

        def quant_target(peaks, target, tol, reduce):
            hits = peaks.irange((target - tol,),
                                (target + tol,))
            if hits:
                return reduce([h[1] for h in hits])
            else:
                return 0

        vector = np.array([quant_target(peaks, t, tol, reduce) for t in targets])
        metadata.append(pd.Series({'x':x,
                                   'y':y,
                                   'z':z,
                                   'file':imzML}))
        vectors.append(vector)
    pixels = pd.DataFrame(metadata)
    if target_names is None:
        target_names = [f'mz_{mz}' for mz in targets]
    vectors = pd.DataFrame(np.array(vectors), columns = target_names)
    pixels = pd.concat([pixels, vectors], axis = 1)
    if drop_uniform_z and len(set(pixels['z'])) == 1:
        del pixels['z']
    return pixels

class PartialPipeline(Pipeline):
    """
    Utility function to generate a `PartialPipeline`

    Arguments:
        steps: a collection of text-transformers

    ```python
    from tokenwiser.pipeline import PartialPipeline
    from tokenwiser.textprep import HyphenTextPrep, Cleaner

    tc = PartialPipeline([('clean', Cleaner()), ('hyp', HyphenTextPrep())])
    data = ["dinosaurhead", "another$$ sentence$$"]
    results = tc.partial_fit(data).transform(data)
    expected = ['di no saur head', 'an other  sen tence']

    assert results == expected
    ```
    """
    def partial_fit(self, X, y=None, **kwargs):
        """
        Fits the components, but allow for batches.
        """
        for name, step in self.steps:
            tags = step.__sklearn_tags__()
            if not hasattr(step, "partial_fit"):
                if tags.requires_fit:
                    raise ValueError(
                        f"Step {name} is a {step} which does not have `.partial_fit` implemented."
                    )
        for name, step in self.steps:
            if hasattr(step, "partial_fit"):
                step.partial_fit(X, y, **kwargs)
            if hasattr(step, "transform"):
                X = step.transform(X)
        return self

class DiskDataset:
    def __init__(self, 
                 path, 
                 chunksize = 1000,
                 shuffle = True,
                 copy = False,
                 seed = 0):
        self.path = os.path.abspath(path)
        self.copy = copy
        self.rng = np.random.default_rng(seed)
        self.chunksize = chunksize
        self.shuffle = shuffle
    
    def _chunk_data(self):
        chunkrange = range(0, self._N_rows, self.chunksize)
        if self.shuffle:
            idxs = self.rng.permuted(range(self._N_rows))
            self.chunks = [idxs[i:i+self.chunksize] for i in chunkrange]
        else:
            self.chunks = [slice(i,i+self.chunksize) for i in chunkrange]
    
    def __iter__(self):
        self._chunk_data()
        return self
    
    def __next__(self):
        if self.chunks:
            return self[self.chunks.pop()]
        else:
            raise StopIteration
    
    def __len__(self):
        self.chunk_data()
        return len(self.chunks)

    def _positive(self, val):
        return val if val >= 0 else self._N_rows + val

    def __getitem__(self, val):
        self._load_metadata()
        if type(val) == slice:
            start = val.start if val.start is not None else 0
            stop = val.stop if val.stop is not None else self._N_rows
            idx = range(self._positive(start), self._positive(stop))
        elif hasattr(val, '__iter__'):
            val = sorted({self._positive(v) for v in val})
            idx = val
        else:
            idx = [self._positive(val)]
            val = [val]
        with h5py.File(self.path, "r", driver=None) as f:
            table = f['data']
            data = table[val, :]
            data = pd.DataFrame(data, 
                                index = idx,
                                columns = self.columns)
        data['file'] = [self._idx_file[int(i)] for i in data['file']]
        return data
    
    def get_file(self, file):
        idx = self._file_idx[file]
        data = pd.concat([self[s] for s in self._idx_rows[idx]], 
                         ignore_index = True)
        return data
    
    def list_files(self):
        return list(self._file_idx.keys())
    
    def _add_file_ranges(self, data):
        #in order to more efficiently retrieve data for specific files
        #we save the data by file in chunks and keep track of the index
        #slices we need to get a file from the combined table
        data['index'] = range(data.shape[0])
        ranges = (data[['file','index']]
                  .groupby('file')['index']
                  .apply(lambda x: slice(min(x)+self._N_rows, 
                                         max(x)+self._N_rows+1))
                  .to_dict()
                  )
        for k,v in ranges.items():
            self._idx_rows[k].append(v)
        del data['index']
        with h5py.File(self.path, "a", driver=None) as f:
            idx_rows = f['idx_rows']
            for idx, rows in self._idx_rows.items():
                if str(idx) in set(f['idx_rows'].keys()):
                    del idx_rows[str(idx)]
                idx_rows[str(idx)] = [[s.start, s.stop] for s in rows]

    
    def _load_metadata(self):
        with h5py.File(self.path, "a", driver=None) as f:
            if f.get('file_idx') is not None:
                file_idx = f['file_idx']
                files = list(file_idx.keys())
                idxs = [i[()] for i in file_idx.values()]
                self._idx_file = {i:f for i,f in zip(idxs, files)}
                self._file_idx = {f:i for i,f in zip(idxs, files)}

                idx_rows = f['idx_rows']
                idxs = list(idx_rows.keys())
                rows = [[slice(s[0],s[1]) for s in r] for r in idx_rows.values()]
                self._idx_rows = {int(i):r for i,r in zip(idxs, rows)}
                
                data = f['data']
                self._N_rows = data.shape[0]
                
                columns = f['columns']
                self.columns = [c.decode() for c in columns[:]]

            else:
                self._idx_file = dict()
                self._file_idx = dict()
                self._idx_rows = defaultdict(lambda:[])
                self._N_rows = 0
                self.columns = None
                f.create_group('file_idx')
                f.create_group('idx_rows')

    def _process_file_info(self, data):
        new_files = list(set(data['file']).difference(self._file_idx.keys()))
        max_idx = max(self._file_idx.values()) if self._file_idx else 0
        new_idxs = [i+max_idx for i in range(len(new_files))]
        self._file_idx.update({f:i for f,i in zip(new_files, new_idxs)})
        data['file'] = [self._file_idx[f] for f in data['file']]
        data = data.sort_values('file')
        with h5py.File(self.path, "a", driver=None) as f:
            file_idx = f['file_idx']
            for file in new_files:
                file_idx[file] = self._file_idx[file]
        return data

    def add_data(self, data):
        self._load_metadata()
        if self.copy:
            data = data.copy()
        if self.columns is None:
            self.columns = list(data.columns)
        self._process_file_info(data)
        self._add_file_ranges(data)
        
        data = data[self.columns].to_numpy()
        
        with h5py.File(self.path, "a", driver=None) as f:
            table = f.get('data')
            if table is None:
                f.create_dataset('data',
                                 data = data,
                                 maxshape=(None, data.shape[1]))
                f.create_dataset('columns',
                                 data = self.columns,
                                 maxshape=len(self.columns))
            else:
                shape = table.shape
                table.resize((data.shape[0]+shape[0],shape[1]))
                table[-data.shape[0]:,:] = data
        self._chunk_data()

