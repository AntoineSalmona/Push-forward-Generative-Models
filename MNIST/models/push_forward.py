import torch
import torch.nn as nn
import torch.nn.functional as F    



cuda = torch.cuda.device_count() > 0
device = torch.device("cuda" if cuda else "cpu")





class Generator(nn.Module):
    """
    The generative network class (i.e the generator for the GAN and the decoder for the VAE)
    Copied and modified from https://github.com/AKASHKADEL/dcgan-mnist/blob/master/networks.py
    """
    def __init__(self,n_latent,n_feature):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(n_latent, n_feature*4, 4, 1, 0, bias=False)
        self.norm1 = nn.BatchNorm2d(n_feature*4)
        self.conv2 = nn.ConvTranspose2d(n_feature*4, n_feature*2, 3, 2, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(n_feature*2)
        self.conv3 = nn.ConvTranspose2d(n_feature*2, n_feature, 4, 2, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(n_feature)
        self.conv_4 = nn.ConvTranspose2d(n_feature,1, 4, 2, 1, bias=False)
    
    def forward(self,x):
        x = F.leaky_relu(self.norm1(self.conv1(x)),0.2)
        x = F.leaky_relu(self.norm2(self.conv2(x)),0.2)
        x = F.leaky_relu(self.norm3(self.conv3(x)),0.2)
        return torch.sigmoid(self.conv_4(x))


class Encoder(nn.Module):
    """
    Copied and modified from https://github.com/AKASHKADEL/dcgan-mnist/blob/master/networks.py
    """
    def __init__(self,n_latent,n_feature):
        super().__init__()
        self.conv1 = nn.Conv2d(1,n_feature, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(n_feature, n_feature * 2, 4, 2, 1, bias=False)   
        self.norm1 = nn.BatchNorm2d(n_feature * 2)
        self.conv3 = nn.Conv2d(n_feature * 2, n_feature * 4, 3, 2, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(n_feature * 4)
        self.conv_mean = nn.Conv2d(n_feature * 4,n_latent, 4, 1, 0, bias=False)
        self.conv_var = nn.Conv2d(n_feature * 4,n_latent, 4, 1, 0, bias=False)

                
    def forward(self,x):
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.norm1(self.conv2(x)),0.2)
        x = F.leaky_relu(self.norm2(self.conv3(x)),0.2)
        mean = self.conv_mean(x)
        log_var = self.conv_var(x)
        epsilon = torch.randn(mean.shape).to(device)
        z = mean + torch.exp(0.5*log_var)*epsilon
        return z, mean, log_var
    


class Discriminator(nn.Module):
    def __init__(self,n_feature,sigmoid=True):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(1,n_feature, 4, 2, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(n_feature, n_feature * 2, 4, 2, 1, bias=False))
        self.norm1 = nn.BatchNorm2d(n_feature * 2)
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(n_feature * 2, n_feature * 4, 3, 2, 1, bias=False))
        self.norm2 = nn.BatchNorm2d(n_feature * 4)
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(n_feature * 4,1, 4, 1, 0, bias=False))
        self.sigmoid = sigmoid

                
    def forward(self,x):
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.norm1(self.conv2(x)),0.2)
        x = F.leaky_relu(self.norm2(self.conv3(x)),0.2)
        if self.sigmoid:
            x = torch.sigmoid(self.conv4(x))
        else:
            x = self.conv4(x)
        return x.view(-1,1).squeeze(1)


class Discriminator2(nn.Module):
    """
    The MLP discriminator works better for the synthetic mixture of Gaussians
    """
    def __init__(self,n_dim,sigmoid=True):
        super().__init__()
        self.fc_1 = nn.utils.spectral_norm(nn.Linear(n_dim,512))
        self.fc_2 = nn.utils.spectral_norm(nn.Linear(512,256))
        self.fc_3 = nn.utils.spectral_norm(nn.Linear(256,128))
        self.fc_4 = nn.utils.spectral_norm(nn.Linear(128,1))
        self.sigmoid = sigmoid


    def forward(self,x):
        x = torch.flatten(x,start_dim=1)
        x = F.leaky_relu(self.fc_1(x),0.2)
        x = F.leaky_relu(self.fc_2(x),0.2)
        x = F.leaky_relu(self.fc_3(x),0.2)
        x = self.fc_4(x)
        if self.sigmoid:
            return torch.sigmoid(x)
        else:
            return x



class GAN(nn.Module):

    def __init__(self,n_latent=100,n_feature=32,sigmoid=True,conv_disc=True):
        super().__init__()
        self.generator = Generator(n_latent,n_feature)
        if conv_disc:
            self.discriminator = Discriminator(n_feature,sigmoid)
        else:
            self.discriminator = Discriminator2(784,sigmoid)
        self.n_latent = n_latent

  
    def forward(self,z):
        x = self.generator(z)
        y = self.discriminator(x)
        return x,y



class VAE(nn.Module):
    
    def __init__(self,n_latent=100,n_feature=32):
        super().__init__()
        self.encoder = Encoder(n_latent,n_feature)
        self.decoder = Generator(n_latent,n_feature)


    def forward(self,x):
        z, mean_e, log_var = self.encoder(x)
        mean_d = self.decoder(z)
        return mean_d, mean_e, log_var