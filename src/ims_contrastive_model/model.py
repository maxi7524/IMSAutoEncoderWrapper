"""
ims_contrastive_model/model.py
--------------------------------
Main class implementing the Contrastive MSI Segmentation Model.
Ties together the architecture, adapter, and training loop.
"""

# basic python
from pathlib import Path
from typing import Union, Optional

# numerical libraries
import numpy as np

# torch
from sympy import hyperexpand
import torch
from torch.utils.data import DataLoader

# local modules
from .architecture import ContrastiveAutoencoder, ContrastiveLoss
from .optimization import suggest_cnn_configuration, train_clr_loop
from .dataloader import IMSPyTorchDataset

# IMS library 
import m2aia as m2

# TODO - by default we assume that we provide parameters for image 

class ImsContrastiveModel:
    def __init__(self, 
                # obligatory
                IMSLoader: IMSPyTorchDataset, 
                latent_dim: int,
                # train parameters
                epochs: int = 10, 
                batch_size: int = 256, 
                lr: float = 1e-3,
                patience: int = 5,
                # hyperparameters
                hyperparameters = None # here put dict # TODO 
            ):
        self._img = IMSLoader
        self._latent_dim = latent_dim
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._patience = patience
        self._hyperparameters = hyperparameters

        # TODO - configure optimal params (optimization module call) 
        params = suggest_cnn_configuration(IMSLoader=IMSLoader, latent_dim=latent_dim, hyperparameters=hyperparameters)

        # TODO - create autoencoder
        model = ContrastiveAutoencoder(**params)

        # TODO - use right device (maybe above )

    def fit(self, itd):
        # TODO train model
        ...

    def load(self, itd):
        # TODO load existing model from path
        ...
        
    def transform(self, itd):
        # TODO encode full image to latent space and return img x latent space 
        ...


    # TODO - additonal like save training data | show model configuration
