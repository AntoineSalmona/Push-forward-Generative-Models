import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F    
from torch.optim import Adam
from torch.autograd import Variable
from scipy.stats import norm


kwargs = {'num_workers': 1, 'pin_memory': True} 
cuda = torch.cuda.device_count() > 0
device = torch.device("cuda" if cuda else "cpu")


class Data(Dataset):
    """
    The abstract class for data 
    """
    def __init__(self,X):
        self.data = X

    
    
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self,idx):
        return torch.FloatTensor(self.data[idx])



def gaussian_mixture(mu,sigma=1,d=1,n=50000):
    """
    Draws n independent samples from a Gaussian mixture (1/2)(N(-m,\sigma^2I_d)+N(m,\sigma^2Id_d))
    """
    X = []
    for k in range(n):
        c = np.random.binomial(1,0.5)
        if c == 0:
            X.append(np.random.multivariate_normal(np.array([mu for k in range(d)]),sigma**2*np.eye(d),1))
        else:
            X.append(np.random.multivariate_normal(np.array([-mu for k in range(d)]),sigma**2*np.eye(d),1))         
    X = np.array(X)[:,0,:]
    return X



def Lipschitz_constant_estimator(network,points=np.linspace(-10.,10.,1000),n=1000):
    """
    Evaluates the Lipschitz constant of the network for the VAE and GAN
    """
    network.eval()
    with torch.no_grad():
        output = network(torch.Tensor(points).reshape(n,1).to(device)).squeeze().cpu().numpy()
        pairwise_diff = np.abs(np.subtract.outer(points,points))
        pairwise_diff_output = np.abs(np.subtract.outer(output,output))
        lipschitz_estimators = np.divide(pairwise_diff_output,pairwise_diff,out=np.zeros_like(pairwise_diff),where=pairwise_diff!=0)
    return np.max(lipschitz_estimators)


def Lipschitz_constant_estimator_score(network,points=np.linspace(-10.,10.,1000),n=1000,sigma=0.):
    """
    Evaluates the Lipschitz constant of the network for the score network
    """
    with torch.no_grad():
        output = network(torch.Tensor(points).reshape(n,1).to(device),torch.FloatTensor(np.array([sigma for k in range(n)])).to(device)).squeeze().cpu().numpy() 
        pairwise_diff = np.abs(np.subtract.outer(points,points))
        pairwise_diff_output = np.abs(np.subtract.outer(output,output))
        lipschitz_estimators = np.divide(pairwise_diff_output,pairwise_diff,out=np.zeros_like(pairwise_diff),where=pairwise_diff!=0)
    return np.max(lipschitz_estimators)




def train_vae(model,Y,n_latent,epochs=100,batch_size=1000,normalize=False,lr=1e-4,save_lip=False):
    
    def elbo(x,mean_d,mean_e,log_var,c,n_latent=2):
        """
        The ELBO loss function
        """
        reconstruction_loss = torch.sum((1./(2*c))*(torch.bmm(x.view(x.shape[0],1,n_latent),x.view(x.shape[0],n_latent,1)) \
                                + torch.bmm(mean_d.view(x.shape[0],1,n_latent),mean_d.view(x.shape[0],n_latent,1))) \
                                - (1./c)*torch.bmm(x.view(x.shape[0],1,n_latent),mean_d.view(x.shape[0],n_latent,1)))
        KLD = 0.5 * torch.sum(mean_e.pow(2) + log_var.exp().pow(2) - log_var - 1)
        return reconstruction_loss.squeeze(-1) + KLD
    
    lip_list = []
    if normalize:
        train_dataset = Data((Y - np.mean(Y))/np.std(Y))
    else:
        train_dataset = Data(Y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    loop = tqdm(range(epochs))
    for epoch in loop:
        train_loss = 0
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            mean_d, mean_e, log_var = model(x)
            loss = elbo(x, mean_d, mean_e, log_var,0.1,n_latent=n_latent)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        loop.set_postfix({'loss': train_loss/len(train_dataset),'Lipschitz constant': Lipschitz_constant_estimator(model.decoder)})
        if save_lip:
            lip_list.append(Lipschitz_constant_estimator(model.generator))
    if save_lip:
        return lip_list


def generate_vae(model,Y,n_gen,n_latent,normalize=False):
    model.eval()
    with torch.no_grad():
        noise = torch.randn((n_gen,n_latent)).to(device)
        gen_data = model(noise)
        if normalize:
            gen_data = np.std(Y)*gen_data.cpu().numpy() + np.mean(Y)
        else:
            gen_data = gen_data.cpu().numpy()
        return gen_data 



def compute_gradient_penalty_2(G,z,L):
    """
    Copied and modified from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
    Calculates the gradient penalty loss 
    """
    fake = Variable(torch.Tensor(z.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    z = z.requires_grad_(True)
    gen_data = G(z)
    gradients = torch.autograd.grad(
        outputs=gen_data,
        inputs=z,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_norm = torch.max(gradient_norm)
    gradient_penalty = (gradient_norm - L)**2
    return gradient_penalty


def train_gan(model,Y,epochs=100,batch_size=1000,normalize=False,lr=1e-4,gradient_penalty=None,save_lip=False):
    if normalize:
        train_dataset = Data((Y - np.mean(Y))/np.std(Y))
    else:
        train_dataset = Data(Y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    criterion = nn.BCELoss() 
    G_optimizer = Adam(model.generator.parameters(),lr=lr)
    D_optimizer = Adam(model.discriminator.parameters(),lr=lr)
    model.train()
    loop = tqdm(range(epochs))
    lip_list = []
    for epoch in loop:   
        train_loss_G = 0       
        train_loss_D = 0          
        for batch_idx, x in enumerate(train_loader):
            valid = Variable(torch.ones(batch_size, 1).to(device),requires_grad=False)
            fake =  Variable(torch.zeros(batch_size, 1).to(device),requires_grad=False)
            x = Variable(x.to(device))
            
            G_optimizer.zero_grad()
            z = Variable(torch.randn(batch_size, model.n_latent).to(device))
            gen_data = model.generator(z)
            if gradient_penalty is None:
                G_loss = criterion(model.discriminator(gen_data),valid)
            else:
                G_loss = criterion(model.discriminator(gen_data),valid) \
                         + (10/gradient_penalty**2)*compute_gradient_penalty_2(model.generator,z,gradient_penalty)
            train_loss_G += G_loss.item()
            G_loss.backward()
            G_optimizer.step()

            D_optimizer.zero_grad()
            real_loss = criterion(model.discriminator(x), valid)
            z = Variable(torch.randn(batch_size, model.n_latent).to(device))
            gen_data = model.generator(z)
            fake_loss = criterion(model.discriminator(gen_data.detach()), fake)
            D_loss = (real_loss + fake_loss)/2                
            train_loss_D += D_loss.item() 
            D_loss.backward()
            D_optimizer.step()
        loop.set_postfix({'Generator loss': train_loss_G,'Discriminator loss': train_loss_D,'Lipschitz constant': Lipschitz_constant_estimator(model.generator)})
        if save_lip:
            lip_list.append(Lipschitz_constant_estimator(model.generator))
    if save_lip:
        return lip_list



def generate_gan(model,Y,n_gen,n_latent,normalize=False):
    model.eval()
    with torch.no_grad():
        noise = torch.randn((n_gen,n_latent)).to(device)
        gen_data = model(noise)
        if normalize:
            gen_data = np.std(Y)*gen_data.cpu().numpy() + np.mean(Y)
        else:
            gen_data = gen_data.cpu().numpy()
        return gen_data







def train_score(model,Y,sigma_list,epochs=100,batch_size=1000,normalize=False,lr=1e-4):
    
    def Fischer_Divergence(x,x_noisy,y,sigma):
        """
        Loss function for training Score network
        """
        return (1./2)*F.mse_loss(y,(x - x_noisy)/sigma**2,reduction='sum')

    def blind_noisify(X,sigma_list,device):
        """
        noisify the data for a random sigma in sigma_list
        """
        sigma = sigma_list[np.random.randint(len(sigma_list))]
        Z = sigma*torch.randn_like(X).to(device)
        sigma = torch.FloatTensor(np.array([sigma for k in range(X.shape[0])])).to(device)
        return X + Z, sigma   

    if normalize:
        X = (Y - np.mean(Y))/np.std(Y)
        train_dataset = Data(X)
    else:
        X = Y
        train_dataset = Data(X)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    loop = tqdm(range(epochs))
    for epoch in loop:
        overall_loss = 0
        for batch_idx, x in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            y, sigma = blind_noisify(x,sigma_list,device)
            x_hat = model(y,sigma)
            loss = sigma[0]*2*Fischer_Divergence(x,y,x_hat,sigma[0])
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        loop.set_postfix({'loss': overall_loss/len(train_dataset),'Lipschitz constant': Lipschitz_constant_estimator_score(model)})
        



def annealed_langevin_dynamic(model,Y,sigma_list,n_latent,n=1000,n_step_each=100,step_lr=0.00002,normalize=True):
    """
    The annealed Langevin dynamic as defined in https://arxiv.org/abs/1907.05600
    """
    model.eval()
    with torch.no_grad():
        x = torch.randn(n,n_latent).to(device)
        for i in tqdm(range(len(sigma_list))):
            sigma_tensor = torch.FloatTensor(np.array([sigma_list[i] for k in range(x.shape[0])])).to(device)
            alpha = step_lr*(sigma_list[i])**2/(sigma_list[-1])**2
            for k in range(n_step_each):
                z = torch.randn(x.shape).to(device)
                x = x + (alpha/2)*model(x,sigma_tensor) + (alpha)**(1./2)*z
    if normalize:
        return np.std(Y)*x.cpu().numpy() + np.mean(Y)
    else:
        return x.cpu().numpy()


def proba(a,b,m=5,sigma=1):
    """
    Compute nu([a,b]) when nu=(1/2)[N(-m,sigma^2)+N(m,sigma^2)]
    """
    return 0.5*(norm.cdf((b-m)/sigma) - norm.cdf((a-m)/sigma)) + 0.5*(norm.cdf((b+m)/sigma) - norm.cdf((a+m)/sigma))


def bound_tv(mu,lip,n_1,grid):
    """
    Grid search on r 
    """
    bound = 0
    for r in np.linspace(0.01,2*mu,1000 + max([0,int((mu - 5)*200)])):
        if len(grid) == 400:
            idx = int(((20 - r/2)/40)*400)
        else:
            idx = int(((10 - r/2)/20)*200)          
        prob_a = np.mean(n_1[0:idx])*(grid[idx]-grid[0])
        alpha = norm.cdf(r/lip+norm.ppf(prob_a)) - prob_a 
        beta = proba(-r/2,r/2,m=mu)
        bound_test = alpha - beta
        if bound_test >= bound:
            bound = bound_test
        else:
            break
    return bound


def bound_kl(mu,lip,n_1,grid):
    """
    Grid search on r
    """
    bound = 0
    for r in np.linspace(0.01,2*mu,1000 + max([0,int((mu - 5)*200)])):
        if len(grid) == 400:
            idx = int(((20 - r/2)/40)*400)
        else:
            idx = int(((10 - r/2)/20)*200)          
        prob_a = np.mean(n_1[0:idx])*(grid[idx]-grid[0])
        alpha = norm.cdf(r/lip+norm.ppf(prob_a)) - prob_a 
        beta = proba(-r/2,r/2,m=mu)
        l = np.log(alpha*(1-beta)) - np.log(beta*(1-alpha))
        bound_test = l*alpha - np.log(1 + (np.exp(l)-1)*beta)
        if bound_test >= bound:
            bound = bound_test
        else:
            break
    return bound 
