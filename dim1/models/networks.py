import torch
import torch.nn as nn
import torch.nn.functional as F    
from dim1.models.layers import get_sinusoidal_positional_embedding
cuda = torch.cuda.device_count() > 0
device = torch.device("cuda" if cuda else "cpu")

class Generator(nn.Module):
    """
    the generative network class (generator for the GAN/encoder for the VAE)
    """
  
    def __init__(self,n_latent=1,n_layers=3,spectral_norm=False):
        super().__init__()
        n = n_layers - 2
        layer_list = [nn.Linear(n_latent,2**7)] \
                        + [nn.Linear(2**(7+k),2**(7+k+1)) for k in range(n)] \
                        + [nn.Linear(2**(7+n),n_latent)]
        if spectral_norm:
            layer_list = [nn.utils.spectral_norm(layer) for layer in layer_list]
        self.layer_list = nn.ModuleList(layer_list)
        self.n_layer = n_layers

    def forward(self,x):
        for k in range(self.n_layer-1):
            x = F.leaky_relu(self.layer_list[k](x),0.2)
        return self.layer_list[-1](x)



class Discriminator(nn.Module):
  
    def __init__(self,n_latent=1,sigmoid=True,n_layers=3):
        super().__init__()
        n = n_layers - 1
        layer_list = [nn.utils.spectral_norm(nn.Linear(2**7,n_latent))] \
                        + [nn.utils.spectral_norm(nn.Linear(2**(7+k+1),2**(7+k))) for k in range(n)] \
                        + [nn.utils.spectral_norm(nn.Linear(n_latent,2**(7+n)))]
        self.layer_list = nn.ModuleList(layer_list[::-1])
        self.sigmoid = sigmoid
        self.n_layer = n_layers + 1


    def forward(self,x):
        for k in range(self.n_layer-1):
            x = F.leaky_relu(self.layer_list[k](x),0.2)
        x = self.layer_list[-1](x)
        if self.sigmoid:
            return torch.sigmoid(x)
        else:
            return x



class Encoder(nn.Module):
  
    def __init__(self,n_latent=1):
        super().__init__()
        self.fc_1 = nn.Linear(n_latent,256)
        self.fc_2 = nn.Linear(256,128)
        self.fc_mean = nn.Linear(128,n_latent)
        self.fc_var = nn.Linear(128,n_latent)


    def forward(self,x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        mean = self.fc_mean(x)
        log_var = self.fc_var(x)
        epsilon = torch.randn(mean.shape).to(device)
        z = mean + torch.exp(0.5*log_var)*epsilon
        return z, mean, log_var



class VAE(nn.Module):

  def __init__(self,n_latent=1,n_layers=3):
    super().__init__()
    self.encoder = Encoder(n_latent)
    self.decoder = Generator(n_latent,n_layers)


  def forward(self,x):
    z, mean_e, log_var = self.encoder(x)
    mean_d = self.decoder(z)
    return mean_d, mean_e, log_var




class GAN(nn.Module):

  def __init__(self,n_latent=1,sigmoid=True,n_layers=3,spectral_norm=False):
    super().__init__()
    self.generator = Generator(n_latent,n_layers,spectral_norm)
    self.discriminator = Discriminator(n_latent,sigmoid,n_layers)
    self.n_latent = n_latent

  
  def forward(self,z):
    x = self.generator(z)
    y = self.discriminator(x)
    return x,y



class ScoreNet(nn.Module):

    def __init__(self,n_latent):
        super().__init__()
        self.fc_1 = nn.Linear(n_latent,96)
        self.fc_2 = nn.Linear(96+64,196)
        self.fc_3 = nn.Linear(196+64,n_latent)
        self.sigma_emb_1 = nn.Linear(16,32)
        self.sigma_emb_2 = nn.Linear(32,64)

    def forward(self,x,sigma):
        sigma = get_sinusoidal_positional_embedding(sigma,16)
        sigma = F.leaky_relu(self.sigma_emb_1(sigma),0.2)
        sigma =  F.leaky_relu(self.sigma_emb_2(sigma),0.2)
        x = F.leaky_relu(self.fc_1(x))
        x = torch.cat([x,sigma],-1)
        x = self.fc_2(x)
        x = torch.cat((x,sigma),-1)
        x = F.leaky_relu(x,0.2)
        return self.fc_3(x)


