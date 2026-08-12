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

some code included in this document (cov_loss) was released under the MIT license
at https://github.com/chieupham/PCAAE
This license can be found in thirdparty_licenses/PCAAE
"""

import torch
import torch.nn as nn

class EncoderModel(nn.Module):
    def __init__(self, vector_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(vector_dim, vector_dim),
            nn.SiLU(),
            nn.Dropout(p = dropout),            
            nn.Linear(vector_dim, 2**6),
            nn.SiLU(),
            nn.Dropout(p = dropout),            
            nn.Linear(2**6, 2**5),
            nn.SiLU(),
            nn.Dropout(p = dropout),            
            nn.Linear(2**5, 1)
            )
 
    def forward(self, x):
        return self.layers(x)

class DecoderModel(nn.Module):
    def __init__(self, vector_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 2**5),
            nn.SiLU(),
            nn.Dropout(p = dropout),            
            nn.Linear(2**5, 2**6),
            nn.SiLU(),
            nn.Dropout(p = dropout),            
            nn.Linear(2**6, vector_dim),
            nn.SiLU(),
            nn.Dropout(p = dropout),            
            nn.Linear(vector_dim, vector_dim)
            )
 
    def forward(self, x):
        return self.layers(x)

class TrainingModel(nn.Module):
    def __init__(self, vector_dim, 
                 frozen_encoders, 
                 encoder = None, 
                 decoder = None, 
                 dropout=0.1):
        super().__init__()
        self.frozen_encoders = nn.ModuleList(frozen_encoders)
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = EncoderModel(vector_dim, dropout)
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = DecoderModel(vector_dim, 
                                        len(frozen_encoders)+1, 
                                        dropout)
    
    def forward(self, x):
        latent_space = []
        for encoder in self.frozen_encoders:
            latent_space.append(encoder(x))
        latent_space.append(self.encoder(x))
        latent_space = torch.concat(latent_space, dim = 1)
        ŷ = self.decoder(latent_space)
        return (ŷ, latent_space)

class TestingModel(nn.Module):
    def __init__(self, encoders, decoder):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.decoder = decoder
    
    def forward(self, x):
        latent_space = []
        for encoder in self.encoders:
            latent_space.append(encoder(x))
        latent_space = torch.concat(latent_space, dim = 1)
        y = self.decoder(latent_space)
        return y

class InferenceModel(nn.Module):
    def __init__(self, encoders):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x):
        latent_space = []
        for encoder in self.encoders:
            latent_space.append(encoder(x))
        latent_space = torch.concat(latent_space, dim = 1)
        return latent_space

def cov_loss(z,step):
    if step>1:
        loss = 0
        for idx in range(step-1):
            loss += ((z[:,idx]*z[:,-1]).mean())**2
        loss = loss/(step-1)
    else:
        loss = torch.zeros_like(z)
    return loss.mean()

mse_loss = nn.MSELoss()

def loss_func(ŷ, y, latent_space = None, step = 0, λ = 0.5):
    mse = λ*mse_loss(ŷ, y)
    if latent_space is not None:
        cov = (1-λ)*cov_loss(latent_space, step)
    else:
        cov = 0
    return mse + cov
