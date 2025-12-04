import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SparseDenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, sparsity_lambda, noise_std, dropout_rate=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.sparsity_lambda = sparsity_lambda
        self.dropout_rate = dropout_rate

        # Encoder with dropout
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_dim))
        layers.append(nn.ReLU())
        # Note: No dropout before latent representation
        self.encoder = nn.Sequential(*layers)

        # Decoder with dropout
        layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate)) 
            prev_dim = h
        layers.append(nn.Linear(prev_dim, input_dim))
        # Note: No dropout on output layer
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        # Add Gaussian noise for denoising
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def loss(self, x, x_hat, z):
        recon_loss = nn.functional.mse_loss(x_hat, x)
        sparsity_loss = self.sparsity_lambda * torch.mean(torch.abs(z))
        return recon_loss + sparsity_loss