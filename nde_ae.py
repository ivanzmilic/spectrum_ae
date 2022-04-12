import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class LayerNorm(nn.Module):
    # https://github.com/pytorch/pytorch/issues/1959#issuecomment-312364139

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        use_layer_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln_norm_layers = nn.ModuleList(
                [LayerNorm(features) for _ in range(2)]
            )

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        if self.use_layer_norm:
            temps = self.ln_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        if self.use_layer_norm:
            temps = self.ln_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""
    def __init__(self, in_features, out_features, hidden_features, context_features=None, 
        num_blocks=2, activation=F.elu, dropout_probability=0.0, use_batch_norm=False, use_layer_norm=False,
        train_loader=None, loss=nn.L1Loss(),
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_mu = nn.Linear(hidden_features, out_features)
        
        if train_loader is not None:
            self.x_std = train_loader.dataset.observations.std((0))
            self.x_mean = train_loader.dataset.observations.mean((0))
        else:
            self.x_std = 1.0
            self.x_mean = 0.0

        self.loss = loss

    def forward_mu(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        return self.output_mu(temps)

    def evaluate(self, x):
        x = (x-self.x_mean)/self.x_std
        return self.forward_mu(x)*self.x_std + self.x_mean

    def forward(self, x):
        x = (x-self.x_mean)/self.x_std
        mu = self.forward_mu(x)
        return self.loss_function(x, mu)

    def loss_function(self, y, dist):
        return self.loss(y,dist)



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from torch.nn import init

class RAE(nn.Module):
    """ Residual Autoencoder (RAE). Works only with 1-dim inputs."""
    def __init__(self, x_size, latent_size, hidden_size = 64, num_blocks=5, train_loader=None,device="cpu"):
        super(RAE, self).__init__()

        self.x_size = x_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.device = device
        
        self.encoder_ = ResidualNet(self.x_size, self.latent_size, hidden_features=self.hidden_size,num_blocks=num_blocks)
        self.decoder_ = ResidualNet(self.latent_size, self.x_size, hidden_features=self.hidden_size,num_blocks=num_blocks)
        
        if train_loader is not None:
            self.x_std = train_loader.dataset.observations.std((0))
            self.x_mean = train_loader.dataset.observations.mean((0))

        else:
            self.x_std = 1.0
            self.x_mean = 0.0


    def encode(self, x):
        # Encoder
        mu_E  = self.encoder_.evaluate(x)
        return mu_E

    def decode(self, z):
        # Decoder
        mu_D  = self.decoder_.evaluate(z)
        return mu_D 

    def forward(self, x):
        x = (x-self.x_mean)/self.x_std
        mu_E = self.encode(x.view(-1, self.x_size))
        mu_D = self.decode(mu_E)
        return mu_D*self.x_std + self.x_mean

