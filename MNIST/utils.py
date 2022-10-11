
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from torch.optim import Adam
import numpy as np
import random


cuda = torch.cuda.device_count() > 0
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} 










def train_score(model,train_dataset,sigma_list,epochs=100,batch_size=128,lr=1e-4):
    
    def blind_noisify_2(X,sigma_list,device):
        """
        noisify data
        """
        sigma = sigma_list[np.random.randint(len(sigma_list))]
        Z = sigma*torch.randn_like(X).to(device)
        sigma = torch.FloatTensor(np.array([sigma for k in range(X.shape[0])])).to(device)
        return X + Z, sigma

    def Fischer_Divergence(x,x_noisy,y,sigma):
        """ 
        The loss function for the score network
        """
        return (1./2)*F.mse_loss(y,(x - x_noisy)/sigma**2,reduction='sum')
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    optimizer = Adam(model.parameters(), lr=lr)
    loop = tqdm(range(epochs))
    for epoch in loop:
        model.train()
        train_loss = 0
        for batch_idx, (x,_) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            y, sigma = blind_noisify_2(x,sigma_list,device)
            x_hat = model(y,sigma)
            loss = sigma[0]**2*Fischer_Divergence(x,y,x_hat,sigma[0])
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            loop.set_postfix({'loss': train_loss/len(train_dataset)})
            loop.set_description(str(int(100*batch_idx*batch_size/len(train_dataset))) + "%")
            loop.refresh()
   

            
def annealed_langevin_dynamic(model,sigma_list,n=1000,n_step_each=100,step_lr=0.00002):
    """
    The annealed Langevin dynamic as defined in https://arxiv.org/abs/1907.05600
    """
    model.eval()
    with torch.no_grad():
        x = torch.randn(n,1,28,28).to(device)
        for i in tqdm(range(len(sigma_list))):
            sigma_tensor = torch.FloatTensor(np.array([sigma_list[i] for k in range(x.shape[0])])).to(device)
            alpha = step_lr*(sigma_list[i])**2/(sigma_list[-1])**2
            for k in range(n_step_each):
                z = torch.randn(x.shape).to(device)
                x = x + (alpha/2)*model(x,sigma_tensor) + (alpha)**(1./2)*z
    return x.cpu()

 



def train_vae(model,train_dataset,epochs=100,batch_size=128,lr=1e-4):
    
    def elbo(x,mean_d,mean_e,log_var,c=0.1):
        """
        The ELBO loss function
        """
        x = x.flatten(start_dim=1)
        mean_d = mean_d.flatten(start_dim=1)
        mean_e = mean_e.flatten(start_dim=1)
        log_var = log_var.flatten(start_dim=1)
        reconstruction_loss = F.binary_cross_entropy(mean_d,x, reduction='sum')
        KLD = 0.5 * torch.sum(mean_e.pow(2) + log_var.exp().pow(2) - log_var - 1)
        return reconstruction_loss + KLD
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True, **kwargs)
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    loop = tqdm(range(epochs))
    for epoch in loop:
        train_loss = 0
        for batch_idx, (x,_) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            mean_d, mean_e, log_var = model(x)
            loss = elbo(x, mean_d, mean_e, log_var,0.1)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            loop.set_postfix({'loss': train_loss/len(train_dataset)})
            loop.set_description(str(int(100*batch_idx*batch_size/len(train_dataset))) + "%")
            loop.refresh()       



def generate_vae(model,n_gen,n_latent):
    model.eval()
    with torch.no_grad():
        noise = torch.randn((n_gen,n_latent,1,1)).to(device)
        gen_data = model(noise)
        gen_data = gen_data.cpu()
        return gen_data 



def train_hinge(model,train_dataset,epochs=100,batch_size=128,lr=1e-4):
    """
    The unconditional hinge version of the adversarial loss as proposed in https://arxiv.org/abs/1705.02894
    """
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    G_optimizer = Adam(model.generator.parameters(),lr=lr)
    D_optimizer = Adam(model.discriminator.parameters(),lr=lr)
    model.train()
    loop =  tqdm(range(epochs))
    for epoch in loop:         
        train_loss_G = 0       
        train_loss_D = 0  
        for batch_idx, (x,_) in enumerate(train_loader):
            x = Variable(x.to(device))
                
            D_optimizer.zero_grad()
            D_out_real = model.discriminator(x)        
            D_loss_real = torch.nn.ReLU()(1.0 - D_out_real).mean()
            z = Variable(torch.randn(batch_size,model.n_latent,1,1).to(device))
            gen_data = model.generator(z)
            D_out_fake = model.discriminator(gen_data.detach())
            D_loss_fake = torch.nn.ReLU()(1.0 + D_out_fake).mean()
            D_loss = D_loss_real + D_loss_fake
            train_loss_D += D_loss.item()
            D_loss.backward()
            D_optimizer.step()
                
            G_optimizer.zero_grad()
            z = Variable(torch.randn(batch_size,model.n_latent,1,1).to(device))
            gen_data = model.generator(z)
            G_loss = -model.discriminator(gen_data).mean()
            train_loss_G += G_loss.item()
            G_loss.backward()
            G_optimizer.step()
            loop.set_postfix({'Generator loss': train_loss_G,'Discriminator loss': train_loss_D})       
            loop.set_description(str(int(100*batch_idx*batch_size/len(train_dataset))) + "%")
            loop.refresh()

    

def generate_gan(model,n_gen,n_latent):
    model.eval()
    with torch.no_grad():
        noise = torch.randn((n_gen,n_latent,1,1)).to(device)
        gen_data = model(noise)
        gen_data = gen_data.cpu()
        return gen_data 



class Data(Dataset):
    """
    Abstract class of Data
    """
    def __init__(self,X):
        self.data = X

    
    
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self,idx):
        return self.data[idx]


def select_modes(data,a=1,b=8):
    """
    Selects the subset of all a and b of MNIST
    """
    new_data = []
    for x in data:
        if x[1] == a:
            new_data.append(x)
        elif x[1] == b:
            new_data.append(x)
    

    return Data(new_data)


def synthetic_gm(data,a=1,b=8,n=10000,sigma=0.15):
    """
    Builds a synthetic Gaussian mixture from 2 reference images
    """
    new_data = []
    og_img = []
    for x in data:
        if x[1] == a:
            y = x
            break
    new_data.append(y)
    og_img.append(y[0])
    for k in range(n):
        new_data.append((y[0]+sigma*torch.randn_like(y[0]),y[1]))
    for x in data:
        if x[1] == b:
            y = x
            break
    new_data.append(y)
    og_img.append(y[0])
    for k in range(n):
        new_data.append((y[0]+sigma*torch.randn_like(y[0]),y[1]))
    random.shuffle(new_data)
    return Data(new_data), og_img