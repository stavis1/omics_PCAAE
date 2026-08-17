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
This license can be found in thirdparty_licenses/scikit-learn-contrib_project-template
"""

from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import validate_data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.exceptions import NotFittedError
from scipy.stats import spearmanr

from omics_PCAAE._model_components import TrainingModel, TestingModel, InferenceModel, Loss
from omics_PCAAE.processing import TorchDataset

torch.use_deterministic_algorithms(True)

class PCAAE(BaseEstimator, TransformerMixin):
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
        Whether to calculate the strength of both the monotonic and nonmonotonic
        correlation between each input feature and each output component.
        As the neural network is a nonlinear transformation these values should 
        not be interpreted as identical to loadings in a PCA or factor analysis.
        However, they do provide similar information about the relationship
        between a component and a feature. These values are stored in the attributes
        `monotonic_loadings_` and `nonmonotonic_loadings_`
        
    warm_start : bool, default=False
        If true re-fitting the model will initialize each iteration with the encoders
        and decoders trained in the previous call to .fit()
    
    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    
    loss_trace_ : dict of shape {str : [float]}
        Each training epoch saves the loss from each minibatch in a list with the epoch
        name as the dictionary key.
    
    monotonic_loadings_ : ndarray of shape (n_features, n_components)
        the Spearman r correlation between each feature and each component

    nonmonotonic_loadings_ : ndarray of shape (n_features, n_components)
        The Pearson correlation coefficient between predicitons from a KNeighbors regression 
        between each feature and each component
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
                 calculate_loadings = False,
                 random_state = 0,
                 warm_start = False,
                 loss_ratio = 0.5):
        self.device = device
        self.N_epochs = N_epochs
        self.N_components = N_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.calculate_loadings = calculate_loadings
        self.random_state = random_state
        self.warm_start = warm_start
        self.loss_ratio = loss_ratio
    
    def _get_training_model(self, N_features, component):
        if self.warm_start and self.training_round_ > 0:
            model = TrainingModel(N_features, 
                                  self._encoders[:component],
                                  self._encoders[component],
                                  self._decoders[component],
                                  self.dropout)
        else:
            model = TrainingModel(N_features, 
                                  self._encoders, 
                                  dropout = self.dropout)
        model = model.to(self.device).to(torch.bfloat16)
        model.train()
        model.frozen_encoders.requires_grad_(False)
        return model
    
    def _get_testing_model(self):
        model = TestingModel(self._encoders, self._decoder)
        model = model.to(self.device).to(torch.bfloat16)
        model.eval()
        return model
    
    def _get_inference_model(self):
        model = InferenceModel(self._encoders)
        model = model.to(self.device).to(torch.bfloat16)
        model.eval()
        return model
    
    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
    
    def _set_random_state(self):
        if hasattr(self.random_state, 'seed'):
            random_state = self.random_state.seed
        else:
            random_state = self.random_state
        torch.manual_seed(random_state)
        g = torch.Generator()
        g.manual_seed(random_state)
        return g
    
    def _get_dataloader(self, X, g):
        if type(X) != np.ndarray:
            X = np.array(X)
        X = TorchDataset(X)
        dataloader = torch.utils.data.DataLoader(X, 
                                                 batch_size=self.batch_size, 
                                                 shuffle=True,
                                                 worker_init_fn= self._seed_worker,
                                                 generator=g)
        return dataloader
    
    def _train_model(self, X):
        g = self._set_random_state()
        if not hasattr(self, 'training_round_'):
            self.training_round_ = 0
        else:
            self.training_round_ += 1
        if self.warm_start:
            if not hasattr(self, '_encoders'):
                self._encoders = []
                self._decoders = []
        else:
            self._encoders = []
        if not hasattr(self, 'loss_trace_'):
            self.loss_trace_ = dict()
        N_samples, N_features = X.shape
        dataloader = self._get_dataloader(X, g)
        loss_func = Loss(self.loss_ratio)
        for component in range(self.N_components):
            model = self._get_training_model(N_features, component)
            optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.LinearLR(optimizer, 
                                                    start_factor=0.01, 
                                                    end_factor=1.0, 
                                                    total_iters=N_samples*self.N_epochs)
            
            for epoch in range(self.N_epochs):
                epoch_title = f'Component {component+1}/{self.N_components}, '
                epoch_title += f'Epoch {epoch+1}/{self.N_epochs}'
                if self.warm_start:
                    epoch_title += f', Training round {self.training_round_+1}'
                self.loss_trace_[epoch_title] = []
                progress_bar = tqdm(dataloader, desc=epoch_title)
                for x in progress_bar:
                    x = x.to(self.device)
                    optimizer.zero_grad()
                    y, latent_space = model(x)
                    loss = loss_func(y.view(-1), 
                                     x.view(-1),
                                     latent_space,
                                     component+1)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    self.loss_trace_[epoch_title].append(loss.item())
                    
                    progress_bar.set_postfix(loss=loss.item())
            
            #save encoder
            if self.warm_start:
                if self.training_round_ == 0:
                    self._encoders.append(model.encoder)
                    self._decoders.append(model.decoder)
                else:
                    self._encoders[component] = model.encoder
                    self._decoders[component] = model.decoder
            else:
                self._encoders.append(model.encoder)
        self._decoder = model.decoder
    
    def _kneighbors_r(self, X, y):
        X = X.reshape(-1,1)
        X = KNeighborsRegressor(n_neighbors = 50).fit(X, y).predict(X)
        return np.corrcoef(X[:,0], y)
    
    def _get_factor_loadings(self, X):
        ls = self.transform(X)
        xcol = range(X.shape[1])
        lcol = range(ls.shape[1])
        self.monotonic_loadings_ = np.array([spearmanr(X[:,f], ls[:,c]) for f in xcol for c in lcol])
        self.nonmonotonic_loadings_ = np.array([self._kneighbors_r(X[:,f], ls[:,c]) for f in xcol for c in lcol])

    def _validate_data(self, *args, **kwargs):
        return validate_data(self, *args, **kwargs)
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, **kwargs):
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
        return self
    
    def partial_fit(self, X, y=None, **kwargs):
        if not self.warm_start:
            raise ValueError('warm_start must be True in order to use .partial_fit()')
        self.fit(X)
    
    def __sklearn_is_fitted__(self):
        if not hasattr(self, '_encoders'):
            raise NotFittedError()
        else:
            return True
    
    def transform(self, X, y = None, **kwargs):
        """Embed observations in the learned latent space.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            The learned latent space of the autoencoder.
        """
        self.__sklearn_is_fitted__()
        X = self._validate_data(X, accept_sparse=False, reset=False)
        model = self._get_inference_model()
        X = TorchDataset(X)
        latent_space = []
        with torch.no_grad():
            progress_bar = tqdm(X, desc = 'Inference')
            for x in progress_bar:
                x = x.unsqueeze(0).to(self.device).to(torch.bfloat16)
                latent_space.append(model(x).view(-1).to(torch.double).numpy(force = True))
        latent_space = np.array(latent_space)
        return latent_space
    
    def get_feature_names_out(self, input_features = None):
        return np.array([f'component {i+1}' for i in range(self.N_components)])
    
    def score(self, X, y = None, **kwargs):
        """Score the reconstruction performance of the fitted model.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        score : float
            The negative log loss of the full encoder-decoder model on the input samples.
            The negative log is returned so that this works within the scikit-learn
            ecosystem of model selection tools such as GridSearchCV. 
            The covariance component of the loss function is not considered for testing.
            This means that the train and test loss are not directly comparible. 
        """
        self.__sklearn_is_fitted__()
        X = self._validate_data(X, accept_sparse=False, reset=False)
        model = self._get_testing_model()
        g = self._set_random_state()
        dataloader = self._get_dataloader(X, g)
        loss_func = Loss(self.loss_ratio)
        loss = []
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc = 'Testing')
            for x in progress_bar:
                x = x.to(self.device)
                x_hat = model(x)
                loss.append(loss_func(x_hat, x).item())
        return -np.log(np.mean(loss))





