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

Some code in this document was released under the BSD 3-clause license
at https://github.com/scikit-learn-contrib/project-template
This is reproduced here:
Copyright (c) 2016, Vighnesh Birodkar and scikit-learn-contrib contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of project-template nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import tqdm
import torch
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context, check_is_fitted
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import KNeighborsRegressor

from omics_PCAAE.model_components import TrainingModel, TestingModel, InferenceModel, loss_func
from omics_PCAAE.data_processing import Dataset

class PCAAE(TransformerMixin, BaseEstimator):
    """An implementation of the Principal Component Analsis AutoEncoder algorithm
    
    This iteratively fits PyTorch based autoencoders with a single dimensional latent space.
    At each iteration the encoder is saved and the decoders of subsequent iterations use the
    concatenated latent spaces of all previous encoders. However, only the encoder for the
    current iteration is subject to training updates. The loss function is a combination of 
    mean squared error loss plus a term that minimizes the covariance between each independent
    autoencoder. The result is a nonlinear dimensionality reduction technique that shares
    some properties with PCA, i.e. the output dimensions are rank ordered by explanatory strength
    and are conditionally independent. 
    
    Parameters
    ----------
    device : str, default='cpu'
        The device used for all pytorch neural network operations, if a CUDA capable GPU
        is available we suggest setting this to 'cuda'
    
    N_epochs : int, default=2
        The number of training epochs per component. The total number of training
        epochs is N_epochs * N_components
        
    N_components : int, default=3
        The number of dimensions that the transformed data will have.
        A new encoder and decoder are trained for each dimension so this has a large
        effect on training time.
    
    learning_rate : float, default=0.002
        This is the learning rate for the AdamW optimizer.
    
    batch_size : int, default=10
        The size of each training batch.
        
    dropout : float, default=0.1
        The per-parameter dropout probability during training.
    
    calculate_loadings : bool, default=False
        calculates two matricies of shape (n_features, n_components):
            monotonic_loadings_ : the Spearman r correlation between each feature
            and each component
        
            nonmonotonic_loadings_ : The Pearson correlation coefficient between 
            predicitons from a KNeighbors regression between each feature and 
            each component
        As the neural network is a nonlinear transformation these values should 
        not be interpreted as identical to loadings in a PCA or factor analysis.
        However, they do provide similar information about the relationship
        between a component and a feature. 
    
    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    """
    
    _parameter_constraints = {
        'device': [str],
        'N_epochs': [int],
        'N_components': [int],
        'learning_rate': [float],
        'batch_size': [int],
        'dropout': [float],
        'calculate_loadings': [bool]
    }

    def __init__(self,
                 device = 'cpu',
                 N_epochs = 2,
                 N_components = 3,
                 learning_rate = 0.002,
                 batch_size = 10,
                 dropout = 0.1,
                 calculate_loadings = False):
        self.device = device
        self.N_epochs = N_epochs
        self.N_components = N_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.calculate_loadings = calculate_loadings
        self._is_fitted_ = False
    
    def _get_training_model(self, X):
        model = TrainingModel(X.shape[1], self.encoders, self.dropout)
        model = model.to(self.device).to(torch.bfloat16)
        model.frozen_encoders.requires_grad_(False)
        return model
    
    def _get_testing_model(self):
        model = TestingModel(self.encoders, self.decoder)
        model = model.to(self.device).to(torch.bfloat16)
        return model
    
    def _get_inference_model(self):
        model = InferenceModel(self.encoders)
        model = model.to(self.device).to(torch.bfloat16)
        return model
    
    def _train_model(self, X):
        self.encoders = []
        comps = range(len(self.N_components))
        epochs = range(len(self.N_epochs))
        self.loss_trace_ = {f'component {c+1}, epoch {e+1}':[] for c in comps for e in epochs}
        X = Dataset(X)
        dataloader = torch.utils.data.DataLoader(X, 
                                                 batch_size=self.batch_size, 
                                                 shuffle=True)
        for component in range(1, self.N_components+1):
            model = self._get_training_model(X)
            optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.LinearLR(optimizer, 
                                                    start_factor=0.01, 
                                                    end_factor=1.0, 
                                                    total_iters=len(X)*self.N_epochs)
            
            #train model
            for epoch in range(self.N_epochs):
                epoch_title = f'Component {component}/{self.N_components}, '
                epoch_title += f'Epoch {epoch+1}/{self.N_epochs}'
                progress_bar = tqdm(dataloader, desc=epoch_title)
                for x in progress_bar:
                    x = x.to(self.device)
    
                    # Forward pass
                    optimizer.zero_grad()
                    ŷ, latent_space = model(x)
                    
                    # Compute loss
                    loss = loss_func(ŷ.view(-1), 
                                   x.view(-1),
                                   latent_space,
                                   component)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    self.loss_trace_[f'component {component+1}, epoch {epoch+1}'].append(loss.item())
                    
                    progress_bar.set_postfix(loss=loss.item())
                avg_loss = np.mean(self.loss_trace[(component, epoch)])
                print(f"{epoch_title}; Avg loss: {avg_loss:.4f}")
            
            #save encoder
            self.encoders.append(model.encoder)
        self.decoder = model.decoder
    
    def _rank_transform(self, X):
        qt = QuantileTransformer(n_quantiles = X.shape[0])
        X = qt.fit_transform()
        return X
    
    def _kneighbors_r(self, X, y):
        X = X.reshape(-1,1)
        X = KNeighborsRegressor(n_neighbors = 50).fit(X, y).predict(X)
        return np.corrcoef(X[:,0], y)
    
    def _get_factor_loadings(self, X):
        ls = self.transform(X)
        ls = self._rank_transform(ls)
        X = self._rank_transform(X)
        xcol = range(X.shape[1])
        lcol = range(ls.shape[1])
        self.monotonic_loadings_ = np.array([np.corrcoef(X[:,f], ls[:,c]) for f in xcol for c in lcol])
        self.nonmonotonic_loadings_ = np.array([self._kneighbors_r(X[:,f], ls[:,c]) for f in xcol for c in lcol])
    
    def __sklearn_is_fitted__(self):
        return 
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._validate_data(X, accept_sparse=False)
        self._train_model(X)
        if self.calculate_loadings:
            self._get_factor_loadings(X)
        self._is_fitted_ = True
        # Return the transformer
        return self
        
    def transform(self, X):
        """A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=False, reset=False)
        model = self._get_inference_model()
        X = Dataset(X)
        latent_space = []
        for x in X:
            x = x.unsqueeze(0).to(self.device).to(torch.bfloat16)
            latent_space.append(model(x).view(-1).to(torch.long).numpy(force = True))
        latent_space = np.array(latent_space)
        return latent_space
    
    def get_feature_names_out(self, input_features = None):
        return np.array([f'component {i+1}' for i in range(self.N_components)])
    
    def _more_tags(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags

