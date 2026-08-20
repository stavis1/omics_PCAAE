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
import tempfile
from collections import defaultdict
from multiprocessing import Pool

import torch
import duckdb as dd
import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser
from scipy.stats import binned_statistic
from sklearn.pipeline import Pipeline
from sortedcontainers import SortedList

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, y = None):
        self.X = torch.tensor(X, dtype = torch.bfloat16)
        if y is not None:
            self.y = torch.tensor(y, dtype = torch.bfloat16)
    
    def __len__(self):
        return self.X.shape[0]
 
    def __getitem__(self, idx):
        data = self.X[idx, :]
        if hasattr(self, 'y'):
            data = (data, self.y[idx])
        return data

def binned_imzML_reader(imzML, 
                        mz_min, 
                        mz_max,
                        ibd = None,
                        bin_width = 0.05,
                        filter_empty_bins = False,
                        drop_uniform_z = True,
                        zlib_intensity = False,
                        zlib_mz = False):
    metadata = []
    vectors = []
    if ibd is None:
        data = ImzMLParser(imzML)
    else:
        data = ImzMLParser(imzML, ibd_file = ibd)
    #pyimzml does not natively handle zlib compressed data
    #so we have to decompress manually
    for idx, (x,y,z) in enumerate(data.coordinates):
        mz_bytes, intensity_bytes = data.get_spectrum_as_string(idx)
        if zlib_intensity:
            intensity_bytes = zlib.decompress(intensity_bytes)
        if zlib_mz:
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
                          drop_uniform_z = True,
                          zlib_intensity = False,
                          zlib_mz = False):
    metadata = []
    vectors = []
    if ibd is None:
        data = ImzMLParser(imzML)
    else:
        data = ImzMLParser(imzML, ibd_file = ibd)
    #pyimzml does not natively handle zlib compressed data
    #so we have to decompress manually
    for idx, (x,y,z) in enumerate(data.coordinates):
        mz_bytes, intensity_bytes = data.get_spectrum_as_string(idx)
        if zlib_intensity:
            intensity_bytes = zlib.decompress(intensity_bytes)
        if zlib_mz:
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

class SDDiterator:
    def __init__(self, SDD):
        self.chunks = list(range(SDD.N_chunks))
        self.SDD = SDD
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.chunks)
    
    def __next__(self):
        if self.chunks:
            return self.SDD[self.chunks.pop()]
        else:
            raise StopIteration

def _parallel_process_chunk(*args, state):
    obj = SparseDiskDataset.__new__(SparseDiskDataset)
    obj.__dict__ = state
    results = obj._process_chunk(*args)
    return results

class SparseDiskDataset:
    def __init__(self, 
                 path, 
                 chunksize = 1000,
                 shuffle = True,
                 copy = False,
                 seed = 0,
                 bin_width = 0.05,
                 zlib_intensity = False,
                 zlib_mz = False,
                 mz_min = 100,
                 mz_max = 2000,
                 max_missing_frac = 0.5,
                 min_intensity = 0,
                 n_jobs = 1):
        if not path.endswith('.parquet'):
            raise ValueError('path should end with .parquet')
        self.path = os.path.abspath(path)
        self.copy = copy
        self.rng = np.random.default_rng(seed)
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.bin_width = bin_width
        self.zlib_intensity = zlib_intensity
        self.zlib_mz = zlib_mz
        self.mz_min = mz_min
        self.mz_max = mz_max
        self.max_missing_frac = max_missing_frac
        self.min_intensity = min_intensity
        self.rng = np.random.default_rng(seed)
        self.imzmls = []
        self.ibds = dict()
        self.N_chunks = 0
        self.n_jobs = n_jobs
    
    def __len__(self):
        return self.N_chunks
    
    def __iter__(self):
        return SDDiterator(self)
    
    def __getitem__(self, idx):
        return dd.query(f'''
                        SELECT * EXCLUDE (chunk)
                        FROM '{self.path}'
                        WHERE chunk = {idx}
                        ''').df()
    
    def _read_imzml_chunk(self,
                    imzml, 
                    idx_start,
                    idx_stop,
                    ibd = None):
        metadata = []
        vectors = []
        if ibd is None:
            data = ImzMLParser(imzml)
        else:
            data = ImzMLParser(imzml, ibd_file = ibd)
        #pyimzml does not natively handle zlib compressed data
        #so we have to decompress manually
        for idx in range(idx_start, idx_stop):
            x,y,z = data.coordinates[idx]
            mz_bytes, intensity_bytes = data.get_spectrum_as_string(idx)
            if self.zlib_intensity:
                intensity_bytes = zlib.decompress(intensity_bytes)
            if self.zlib_mz:
                mz_bytes = zlib.decompress(mz_bytes)
            intensity = np.frombuffer(intensity_bytes, 
                                      dtype=data.intensityPrecision)
            mz = np.frombuffer(mz_bytes, 
                               dtype=data.mzPrecision)
            vector = binned_statistic(mz,
                                      intensity, 
                                      statistic = 'sum',
                                      bins = int((self.mz_max - self.mz_min)/self.bin_width),
                                      range = (self.mz_min, self.mz_max))
            metadata.append(pd.Series({'x':x,
                                       'y':y,
                                       'z':z,
                                       'file':imzml}))
            vectors.append(vector.statistic)
        pixels = pd.DataFrame(metadata)
        vectors = np.array(vectors)
        vector_mask = np.any(vectors > self.min_intensity, axis = 0)
        vectors = vectors[:,vector_mask]
        vectors[~np.isfinite(vectors)] = 0
        mz_cols = vector.bin_edges[:-1][vector_mask]
        mz_cols += self.bin_width/2
        mz_cols = [f'mz_{mz}' for mz in mz_cols]
        vectors = pd.DataFrame(vectors, columns = mz_cols)
        pixels = pd.concat([pixels, vectors], axis = 1)
        if len(set(pixels['z'])) == 1:
            del pixels['z']
        return pixels
    
    
    def _process_chunk(self, 
                       tempdir, 
                       imzml, 
                       idx_start, 
                       idx_stop):
        pixels = self._read_imzml_chunk(imzml,
                                        idx_start, 
                                        idx_stop,
                                        self.ibds[imzml])
        N_pixels = pixels.shape[0]
        quantcols = [c for c in pixels.columns if c not in self.metadata]
        col_counts = pixels[quantcols].apply(lambda x: np.sum(x > self.min_intensity)).to_dict()
        path = os.path.join(tempdir, str(hash((imzml, idx_start, idx_stop))) + '.parquet')
        pixels.to_parquet(path, engine='fastparquet', index = False)
        return (col_counts, N_pixels, path)
    
    def read_data(self, imzml_list, ibd_list = []):
        col_counts = defaultdict(lambda:0)
        self.imzmls.extend(imzml_list)
        ibds = defaultdict(lambda:None, {mz:bd for mz,bd in zip(imzml_list, ibd_list)})
        self.ibds.update(ibds)
        N_pixels = 0
        parquets = []
        
        self.metadata = ['x', 'y', 'z', 'file']
        with tempfile.TemporaryDirectory() as tempdir:
            jobs = []
            for imzml in imzml_list:
                data = ImzMLParser(imzml)
                n_spectra = len(data.coordinates)
                breakpoints = list(range(0, n_spectra, self.chunksize))
                breakpoints += [n_spectra]
                jobs.extend([(tempdir, imzml, breakpoints[i-1], breakpoints[i]) for i in range(1, len(breakpoints))])
            if self.n_jobs == 1:
                results = [self._process_chunk(*j) for j in jobs]
            else:
                with Pool(self.n_jobs) as p:
                    jobs = [(j, self.__dict__) for j in jobs]
                    results = p.starmap(_parallel_process_chunk, jobs)
            for result in results:
                counts, pixel_count, path = result
                for k,v in counts:
                    col_counts[k] += v
                N_pixels += pixel_count
                parquets.append(path)
            self.quantcols = [k for k,v in col_counts.items() if 1-(v/N_pixels) < self.max_missing_frac]
            self.N_chunks = int(np.ceil(N_pixels/self.chunksize))
            col_list = self.metadata + self.quantcols

            for parquet in parquets:
                data = pd.read_parquet(parquet, engine='pyarrow')
                if not 'z' in data.columns:
                    self.metadata.remove('z')
                    col_list.remove('z')
                missing_cols = list(set(col_list).difference(data.columns))
                missing = np.full((data.shape[0], len(missing_cols)), 0)
                missing = pd.DataFrame(missing, 
                                       index = data.index, 
                                       columns = missing_cols)
                data = pd.concat([data, missing])
                data = data[col_list]
                mask = np.any(data[self.quantcols] > self.min_intensity, axis = 1)
                data = data[mask]
                data['chunk'] = self.rng.choice(range(self.N_chunks), data.shape[0])
                if os.path.exists(self.path):
                    data.to_parquet(self.path, engine='fastparquet', append=True)
                else:
                    data.to_parquet(self.path, engine='fastparquet')

    def get_file(self, file):
        return dd.query(f'''
                        SELECT * EXCLUDE (chunk)
                        FROM '{self.path}'
                        WHERE file = '{file}'
                        ''').df()
    
    def load_from_file(self, ibd_map = None):
        self.ibds = defaultdict(lambda: None)
        if ibd_map is not None:
            self.ibds.update(ibd_map)
        pixels = self[0]
        self.metadata = ['x', 'y', 'z', 'file']
        if not 'z' in pixels.columns:
            self.metadata.remove('z')
        self.quantcols = [c for c in pixels.columns if not c in self.metadata]
        chunks = dd.query(f'''
                          SELECT DISTINCT chunk
                          FROM '{self.path}'
                          ''').df()
        self.N_chunks = chunks.shape[0]
        files = dd.query(f'''
                         SELECT DISTINCT file
                         FROM '{self.path}'
                         ''').df()
        self.imzmls = list(files['file'])
