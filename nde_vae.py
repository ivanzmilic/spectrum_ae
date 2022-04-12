import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import random


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _strictly_tril_size(n):
    """Unique elements outside the diagonal
    """
    return n * (n-1) // 2


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class create_Coder(nn.Module):
    """Creates a encoder or a decoder according to input parameters
    """
    def __init__(self, input_size, output_size, hidden_size=[64,64], activation=nn.ELU(), full_cov=True, methodfullcov="scale_tril"):
        super(create_Coder, self).__init__()
        self.activation = activation
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.epsrho = 1e-2
        self.full_cov = full_cov
        self.methodfullcov = methodfullcov

        # Input layers
        coder_net_list = []
        coder_net_list.append( nn.Linear(input_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            coder_net_list.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.coder_net_list = nn.ModuleList(coder_net_list)

        # Output layers
        self.output_mu = nn.Linear(hidden_size[-1], output_size)
        self.output_logvar = nn.Linear(hidden_size[-1], output_size)
        
        if self.full_cov is True:
            self.output_chol_net = nn.Linear(hidden_size[-1], _strictly_tril_size(output_size))
            self.lt_indices = torch.tril_indices(output_size, output_size, -1)
    
    def forward(self, xx):
        
        # Pass through input layers
        for ii, ilayer in enumerate(self.coder_net_list):
            xx = self.activation(ilayer(xx))

        # Pass through output layers
        mu = self.output_mu(xx)
        logvar = -self.elu(self.output_logvar(xx))

        if self.full_cov is False:
            dist = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=torch.exp(logvar.mul(0.5))),1)

            return mu, logvar, dist
        else:

            if self.methodfullcov == "scale_tril":

                rho = self.output_chol_net(xx)

                diagA = torch.exp(logvar.mul(0.5))
                diagB = torch.exp(logvar)
                diag = torch.diag_embed(diagA)
                chol = torch.zeros_like(diag)
                chol[..., self.lt_indices[0], self.lt_indices[1]] = rho
                chol = chol + diag
                dist = torch.distributions.MultivariateNormal(loc=mu, scale_tril=chol)

            elif self.methodfullcov == "covariance_matrix":
                rho = self.tanh(self.output_chol_net(xx))* (1.0- self.epsrho)
               
                diagA = torch.exp(logvar.mul(0.5))
                diagB = torch.exp(logvar)
                diag = torch.diag_embed(diagB)
                chol = torch.zeros_like(diag)

                for jj in range(self.lt_indices.shape[1]):
                    chol[..., self.lt_indices[0][jj], self.lt_indices[1][jj]] = rho[:,jj]*diagA[:,self.lt_indices[0][jj]]*diagA[:,self.lt_indices[1][jj]]
                    chol[..., self.lt_indices[1][jj], self.lt_indices[0][jj]] = rho[:,jj]*diagA[:,self.lt_indices[0][jj]]*diagA[:,self.lt_indices[1][jj]]
                chol = chol + diag
                # torch.cholesky(chol)
                dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=chol)
                 
            return mu, [logvar, rho], dist


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class create_Coder_Simple(nn.Module):
    """Creates a encoder or a decoder according to input parameters
    """
    def __init__(self, input_size, output_size, hidden_size=[64,64], activation=nn.ELU()):
        super(create_Coder_Simple, self).__init__()
        self.activation = activation

        # Input layers
        coder_net_list = []
        coder_net_list.append( nn.Linear(input_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            coder_net_list.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.coder_net_list = nn.ModuleList(coder_net_list)

        # Output layers
        self.output_mu = nn.Linear(hidden_size[-1], output_size)
        
    
    def forward(self, xx):
        
        # Pass through input layers
        for ii, ilayer in enumerate(self.coder_net_list):
            xx = self.activation(ilayer(xx))

        # Pass through output layers
        mu = self.output_mu(xx)

        return mu





# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class VAE(nn.Module):
    """ Variational Autoencoder (VAE) WITH REPARAMETRIZATION!
    """
    def __init__(self, x_size, latent_size, full_cov=True, hidden_size = [64, 64, 64, 64, 64], train_loader=None, methodfullcov="scale_tril"):
        super(VAE, self).__init__()
        self.x_size = x_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.full_cov = full_cov
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.methodfullcov = methodfullcov
        
        self.encoder_ = create_Coder(self.x_size, self.latent_size, full_cov=False, hidden_size=self.hidden_size)
        
        self.decoder_ = create_Coder(self.latent_size, self.x_size, full_cov=self.full_cov, 
            hidden_size=self.hidden_size, methodfullcov=self.methodfullcov)
        
        if self.full_cov is True: self.lt_indices = self.decoder_.lt_indices

        if train_loader is not None:
            self.x_std = train_loader.dataset.observations.std((0,1))
        else:
            self.x_std = 1.0


    def encode(self, x): # x
        # Encoder
        mu_E, logvar_E, dist_E  = self.encoder_(x)
        return mu_E, logvar_E, dist_E


    def decode(self, z):
        # Decoder
        mu_D, logvar_D, dist_D = self.decoder_(z)
        return mu_D, logvar_D, dist_D

    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar.mul(0.5))
        eps = torch.empty_like(std).normal_()
        eps.to(self.device)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        x = x/self.x_std
        mu_E, logvar_E, dist_E = self.encode(x.view(-1, self.x_size))
        z = self.reparametrize(mu_E, logvar_E)
        mu_D, logvar_D, dist_D = self.decode(z)
        loss  = self.loss_function(x, dist_D, mu_E, logvar_E)
        return loss

    def loss_function(self, x, dist, mu_E, logvar_E, kld_weight=1.0):
        # Reconstruction loss
        neg_logp_total = dist.log_prob(x)
        rec = -torch.mean(neg_logp_total)

        # KL divergence loss
        kl_div = -1 - logvar_E + mu_E ** 2 + logvar_E.exp()
        kld = torch.mean(kl_div.sum(dim=1) / 2.0)
        return rec+kld_weight*kld

    def sample(self, x):
        # Forward pass
        x = x/self.x_std
        mu_E, logvar_E, dist_E   = self.encode(x.view(-1, self.x_size))
        z = self.reparametrize(mu_E, logvar_E)
        mu_D, logvar_D, dist_D = self.decode(z)
        return dist.sample()*self.x_std






# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class VAE2(nn.Module):
    """ Variational Autoencoder (VAE) WITHOUT REPARAMETRIZATION!
    """
    def __init__(self, x_size, latent_size, full_cov=True, hidden_size = [64, 64, 64, 64, 64], train_loader=None, methodfullcov="scale_tril"):
        super(VAE2, self).__init__()
        self.x_size = x_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.full_cov = full_cov
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.methodfullcov = methodfullcov
        
        self.encoder_ = create_Coder_Simple(self.x_size, self.latent_size, hidden_size=self.hidden_size)
        
        self.decoder_ = create_Coder(self.latent_size, self.x_size, full_cov=self.full_cov, 
            hidden_size=self.hidden_size, methodfullcov=self.methodfullcov)
        
        if self.full_cov is True: self.lt_indices = self.decoder_.lt_indices

        if train_loader is not None:
            self.x_std = train_loader.dataset.observations.std((0,1))
        else:
            self.x_std = 1.0


    def encode(self, x): # x
        # Encoder
        mu_E  = self.encoder_(x)
        return mu_E


    def decode(self, z):
        # Decoder
        mu_D, logvar_D, dist = self.decoder_(z)
        return mu_D, logvar_D, dist

    def forward(self, x):
        x = x/self.x_std
        mu_E = self.encode(x.view(-1, self.x_size))
        mu_D, logvar_D, dist = self.decode(mu_E)
        loss  = self.loss_function(x, dist)
        return loss

    def loss_function(self, x, dist):
        # Reconstruction loss
        neg_logp_total = dist.log_prob(x)
        rec = -torch.mean(neg_logp_total)
        return rec

    def sample(self, x):
        # Forward pass
        x = x/self.x_std
        mu_E  = self.encode(x.view(-1, self.x_size))
        mu_D, logvar_D, dist = self.decode(mu_E)
        return dist.sample()*self.x_std

