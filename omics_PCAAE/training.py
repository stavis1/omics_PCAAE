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

from omics_PCAAE.model import TrainingModel, loss_func

def train_model(data, 
                device = None,
                N_epochs = 2,
                N_components = 5,
                learning_rate = 0.002,
                batch_size = 10):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    encoders = []
    for component in range(1, N_components+1):
        model = TrainingModel(data[0].shape[0], encoders)
        model = model.to(device).to(torch.bfloat16)
        model.frozen_encoders.requires_grad_(False)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.LinearLR(optimizer, 
                                                start_factor=0.01, 
                                                end_factor=1.0, 
                                                total_iters=len(data)*N_epochs)
        
        #train model
        loss_trace = []
        for epoch in range(N_epochs):
            epoch_title = f"Component {component}/{N_components}, Epoch {epoch+1}/{N_epochs}"
            progress_bar = tqdm(dataloader, desc=epoch_title)
            for x in progress_bar:
                x = x.to(device)

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
                loss_trace.append(loss.item())
                
                progress_bar.set_postfix(loss=loss.item())
            avg_loss = np.mean(loss_trace)
            print(f"{epoch_title}; Avg loss: {avg_loss:.4f}")
        
        #save encoder
        encoders.append(model.encoder)
    return (encoders, model.decoder)



