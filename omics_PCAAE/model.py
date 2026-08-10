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
"""

import tqdm
import torch
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context

from omics_PCAAE.model_components import TrainingModel, TestingModel, InferenceModel, loss_func

class PCAAE(TransformerMixin, BaseEstimator):
    """An example transformer that returns the element-wise square root.
    
    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.
    
    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    
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
        'dropout': [float]
    }

    def __init__(self,
                device = 'cpu',
                N_epochs = 2,
                N_components = 5,
                learning_rate = 0.002,
                batch_size = 10,
                dropout = 0.1):
        self.device = device
        self.N_epochs = N_epochs
        self.N_components = N_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
    
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
        self.loss_trace_ = {f'component {c+1}, epoch {e+1}':[] for c in range(len(self.N_components)) for e in range(len(self.N_epochs))}
        dataloader = torch.utils.X.DataLoader(X, 
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
                epoch_title = f"Component {component}/{self.N_components}, Epoch {epoch+1}/{self.N_epochs}"
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
    
    def _get_factor_loadings(self, X):
        model = self._get_inference_model()
        pass

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
        self._get_factor_loadings(X)
        # Return the transformer
        return self
    
    def _more_tags(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags
