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

import warnings
from copy import copy

from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np
from sortedcontainers import SortedList
from sklearn.base import (BaseEstimator, 
                          ClassifierMixin, 
                          _fit_context, 
                          MetaEstimatorMixin, 
                          RegressorMixin,
                          TransformerMixin,
                          clone)
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data, check_X_y
from sklearn.neighbors import KNeighborsRegressor
from sklearn.exceptions import NotFittedError, ConvergenceWarning
from scipy.stats import spearmanr, pearsonr

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
        The Pearson correlation coefficient between predictions from a KNeighbors regression 
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
        return pearsonr(X[:,0], y).statistic
    
    def _get_factor_loadings(self, X):
        ls = self.transform(X)
        xcol = range(X.shape[1])
        lcol = range(ls.shape[1])
        self.monotonic_loadings_ = np.array([spearmanr(X[:,f], ls[:,c]).statistic for f in xcol for c in lcol])
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

class SpyEM(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """An implementation of the Spy-EM algorithm for positive-unlabeled learning
    
    
    Parameters
    ----------
    esitmator : Scikit-Learn regressor
        The underlying regression model to be fit at each iteration
    
    N_iter : int, default=5
        The number of training iterations to complete. If N_iter=1 iterations can be controlled manually 
        with the .iterate() method, this is useful for out-of-core learning with .partial_fit()
    
    FNR : float, default=0.05
        The target false negative rate used for setting the prediction threshold.
        If FNR=0 then no FNR control is performed and the threshold is set to 0.5.
    
    spy_frac : float, default=0
        The fraction of labeled positive examples to treat as negative for setting the FNR threshold.
        Should be between 0 and 1. If spy_frac=0 then no FNR control is performed and the threshold is set to 0.5. 
        
    calculate_loadings : bool, default=False
        Whether to calculate the strength of both the monotonic and nonmonotonic
        correlation between each input feature and the model score.
        
    random_state : int or RandomState instance, default=0
        The seed of the pseudo random number generator that selects the spy subset. 
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    estimator_ : Scikit-Learn regressor
        The fitted regression model.
    
    threshold_ : float
        The decision threshold used for FDR control.
    
    monotonic_loadings_ : ndarray of shape (n_features)
        the Spearman r correlation between each feature and the positive probability

    nonmonotonic_loadings_ : ndarray of shape (n_features)
        The Pearson correlation coefficient between predictions from a 
        KNeighbors regression and the positive probability
    """
    
    _parameter_constraints = {
        'estimator': [RegressorMixin],
        'N_iter': [int],
        'FNR' : [float, int],
        'spy_frac' : [float, int],
        'calculate_loadings' : [bool],
        'random_state' : [int],
    }

    def __init__(self,
                 estimator,
                 N_iter=5,
                 FNR=0.05,
                 spy_frac=0,
                 calculate_loadings=False,
                 random_state=0):
        self.estimator = estimator
        self.N_iter = N_iter
        self.FNR = FNR
        self.spy_frac = spy_frac
        self.calculate_loadings = calculate_loadings
        self.random_state = random_state
    
    def __sklearn_is_fitted__(self):
        if not hasattr(self, 'estimator_'):
            raise NotFittedError()
            
    def _kneighbors_r(self, X, y):
        X = X.reshape(-1,1)
        X = KNeighborsRegressor(n_neighbors = 50).fit(X, y).predict(X)
        return pearsonr(X, y).statistic
    
    def get_feature_loadings(self, X):
        score = self.decision_function(X)
        self.monotonic_loadings_ = np.array([spearmanr(X[:,f], score).statistic for f in range(X.shape[1])])
        self.nonmonotonic_loadings_ = np.array([self._kneighbors_r(X[:,f], score) for f in range(X.shape[1])])
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, *args, **kwargs):
        X, y = check_X_y(X, y, accept_sparse=False, force_writeable=True)
        y = np.bool(y)
        random_state = check_random_state(self.random_state)
        
        #set a subset of positive labels to zero for an independent estimate of the FDR threshold
        spy = random_state.random_sample(y.shape) < self.spy_frac
        spy = np.logical_and(spy, y)
        _y = copy(y)
        _y[spy] = 0
        self._spy = spy
        
        #iteratively re-fit the estimator and update the labels
        for i in range(self.N_iter):
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, _y, *args, **kwargs)
            _y = self.decision_function(X)
            _y[y] = 1
        
        #control the expected FNR using the spy subset
        idx = np.logical_not(y)
        idx = np.logical_or(idx, spy)
        self.fit_fnr(X[idx, :], y[idx])
        
        #calculate relationships between features and predictions
        if self.calculate_loadings:
            self.get_feature_loadings(X)
        return self
    
    def partial_fit(self, X, y, *args, **kwargs):
        X, y = check_X_y(X, y, accept_sparse=False, force_writeable=True)
        y = np.bool(y)
        if not hasattr(self, 'estimator_') or self.estimator_ is None:
            self.estimator_ = clone(self.estimator)
        if hasattr(self, '_old_estimator'):
            _y = self._old_estimator.predict(X)
            _y = np.clip(_y, a_min = 0, a_max = 1)
            _y[y] = 1            
        else:
            _y = y
        self.estimator_.partial_fit(X, _y, *args, **kwargs)
        return self
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit_fnr(self, X, y, *args, **kwargs):
        self.__sklearn_is_fitted__()
        if self.FNR > 0:
            X, y = check_X_y(X, y, accept_sparse=False, force_writeable=True)
            scale = 1/self.spy_frac if self.spy_frac > 0 else 1
            preds = self.decision_function(X)
            P = SortedList(preds[np.bool(y)])
            U = SortedList(preds[~np.bool(y)])
            thresholds = sorted(preds, reverse = True)
            for threshold in thresholds:
                N_U = U.bisect_right(threshold)
                N_P = P.bisect_right(threshold)*scale
                FNR = N_P/N_U if N_U else 1
                if FNR < self.FNR:
                    self.threshold_ = threshold
                    break
            if not hasattr(self, 'threshold_'):
                self.threshold_ = 0.5
                message = 'No threshold found that controls the FNR at the desired level. This may be due to either poor classifier performance or too few spy samples. falling back on default threshold of 0.5'
                warnings.warn(message, ConvergenceWarning)
        else:
            self.threshold_ = 0.5
        return self
        
    def iterate(self):
        if hasattr(self, 'estimator_'):
            self._old_estimator = self.estimator_
            self.estimator_ = None
    
    def decision_function(self, X, *args, **kwargs):
        preds = self.estimator_.predict(X)
        preds = np.clip(preds, a_min = 0, a_max = 1)
        return preds

    def predict_proba(self, X, *args, **kwargs):
        self.__sklearn_is_fitted__()
        X = validate_data(self, X)
        preds = self.decision_function(X)
        return np.array([preds, 1-preds]).T

    def predict(self, X, *args, **kwargs):
        self.__sklearn_is_fitted__()
        X = validate_data(self, X)
        preds = self.decision_function(X)
        return np.int32(preds >= self.threshold_)



