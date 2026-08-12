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

import torch
import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser
from scipy.stats import binned_statistic
from sklearn.pipeline import Pipeline

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype = torch.bfloat16)
    
    def __len__(self):
        return self.X.shape[0]
 
    def __getitem__(self, idx):
        return self.X[idx, :]

def binned_imzML_reader(imzML_list, 
                        mz_min, 
                        mz_max,
                        ibd_list = None,
                        bin_width = 0.05,
                        filter_empty_bins = False,
                        drop_uniform_z = True):
    if ibd_list is not None:
        ibds = {m:i for m,i in zip(imzML_list, ibd_list)}
    metadata = []
    vectors = []
    for imzML in imzML_list:
        if ibd_list is None:
            data = ImzMLParser(imzML)
        else:
            data = ImzMLParser(imzML, ibd_file = ibds[imzML])
        for idx, (x,y,z) in enumerate(data.coordinates):
            mz, intensity = data.getspectrum(idx)
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
    mz_cols = [f'mz_{mz}' for mz in mz_cols]        
    pixels[mz_cols] = vectors
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
    def partial_fit(self, X, y=None, classes=None, **kwargs):
        """
        Fits the components, but allow for batches.
        """
        for name, step in self.steps:
            if not hasattr(step, "partial_fit"):
                raise ValueError(
                    f"Step {name} is a {step} which does not have `.partial_fit` implemented."
                )
        for name, step in self.steps:
            if hasattr(step, "predict"):
                step.partial_fit(X, y, classes=classes, **kwargs)
            else:
                step.partial_fit(X, y)
            if hasattr(step, "transform"):
                X = step.transform(X)
        return self




