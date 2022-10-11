import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F    
from torch.optim import Adam

"""
Pytorch transcription of https://github.com/mducoffe/Learning-Wasserstein-Embeddings
"""


cuda = torch.cuda.device_count() > 0
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} 


train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
    
)


test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Copied from https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
    """
    def __init__(self, patience=3, verbose=False, delta=0, path='MNIST/DWE/DWE.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class EncoderDWE(nn.Module):

        
    def __init__(self,embedding_size=50):
        super().__init__()
        self.conv1 = nn.Conv2d(1,20,3)
        self.conv2 = nn.Conv2d(20,5,5)
        self.fc_1 = nn.Linear(2420,100)
        self.fc_2 = nn.Linear(100,embedding_size)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(F.relu(self.conv2(x)),start_dim=1)
        x = self.fc_1(x)
        return self.fc_2(x)



class DecoderDWE(nn.Module):
    
    def __init__(self,embedding_size=50):
        super().__init__()
        self.fc_1 = nn.Linear(50,100)
        self.fc_2 = nn.Linear(100,5*28*28)
        self.conv1 = nn.Conv2d(5,10,5,padding=2)
        self.conv2 = nn.Conv2d(10,1,3,padding=1)
        
    def forward(self,x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x)).view(x.shape[0],5,28,28)
        x = F.relu(self.conv1(x))
        x = torch.flatten(self.conv2(x),start_dim=1)
        return torch.log_softmax(x,dim=-1).view(x.shape[0],1,28,28)


class DWE(nn.Module):
    def __init__(self,embedding_size=50):
        super().__init__()
        self.encoder = EncoderDWE(embedding_size)
        self.decoder = DecoderDWE(embedding_size)

    def emd(self,x,y):
        z_1 = self.encoder(x)
        z_2 = self.encoder(y)
        return torch.norm(z_1-z_2,dim=-1)**2, z_1, z_2

    def sparcity_constraint(self,y):
        return torch.mean(torch.sum(torch.sqrt(y + 1e-7),dim=(1,2,3)))

    def KLdiv(self,x,y):
        x = torch.clip(x,min=1e-7,max=1)
        y = torch.clip(y,min=1e-7,max=1)
        return torch.mean(torch.sum(x + torch.log(x/y),dim=(1,2,3)))

    def forward(self,x_1,x_2):
        distance, z_1, z_2 = self.emd(x_1,x_2)
        return distance, self.decoder(z_1), self.decoder(z_2), 

    
        
class Data_DWE(Dataset):
    def __init__(self,X,file='MNIST/DWE/emd_mnist.npy',n_samples=700000,start=0,dataset='mnist'):
        self.data = X
        emd_data = np.load(file)
        self.idx_1 = emd_data[0][start:start+n_samples]
        self.idx_2 = emd_data[1][start:start+n_samples]
        self.emd = emd_data[2][start:start+n_samples]
        self.dataset = dataset



    def __len__(self):
        return len(self.emd)
  
    def __getitem__(self,idx):
        if self.dataset == 'mnist':
            x_1 = self.data[int(self.idx_1[idx])][0].double().numpy()
            x_2 = self.data[int(self.idx_2[idx])][0].double().numpy()
        else:
            x_1 = np.float64(self.data[int(self.idx_1[idx])])
            x_2 = np.float64(self.data[int(self.idx_2[idx])])
            x_1 /= 255
            x_2 /= 255
        x_1 /= np.sum(x_1)
        x_2 /= np.sum(x_2)

        return torch.FloatTensor(x_1).view(1,28,28), torch.FloatTensor(x_2).view(1,28,28), np.float32(self.emd[idx])


        
    


def train_DWE(model,train_dataset,valid_dataset,epochs=100,batch_size=100,lr=1e-4,early_stopping=True):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,**kwargs)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    if early_stopping:
        early_stop= EarlyStopping(verbose=False)
    optimizer = Adam(model.parameters(),lr=lr)
    loop = tqdm(range(epochs))
    Kl_loss = nn.KLDivLoss(reduction="batchmean")
    mse = torch.nn.MSELoss()
    for epoch in loop:
        train_loss = 0
        model.train()
        for batch_idx, (x_1,x_2,w) in enumerate(train_loader):
            optimizer.zero_grad()
            x_1 = Variable(x_1).to(device)
            x_2 = Variable(x_2).to(device)
            w = Variable(w).to(device)
            distance, x_1_hat, x_2_hat = model(x_1,x_2)
            loss = mse(distance,w) + 1e1*Kl_loss(x_1_hat,x_1) + 1e1*Kl_loss(x_2_hat,x_2) + 1e-3*model.sparcity_constraint(torch.exp(x_1_hat)) + 1e-3*model.sparcity_constraint(torch.exp(x_2_hat))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()     
            loop.set_description(str(int(100*batch_idx*batch_size/len(train_dataset))) + "%")
            loop.refresh()
        valid_loss = 0
        model.eval()
        for batch_idx, (x_1,x_2,w) in enumerate(valid_loader):
            x_1 = x_1.to(device)
            x_2 = x_2.to(device)
            w = w.to(device)
            distance, x_1_hat, x_2_hat = model(x_1,x_2)
            loss = mse(distance,w) + 1e1*Kl_loss(x_1_hat,x_1) + 1e1*Kl_loss(x_2_hat,x_2) + 1e-3*model.sparcity_constraint(torch.exp(x_1_hat)) + 1e-3*model.sparcity_constraint(torch.exp(x_2_hat))
            valid_loss += loss.item()
        loop.set_postfix({'train loss': train_loss/len(train_dataset),'valid loss': valid_loss/len(valid_dataset)})
        #torch.save(model.state_dict(),'save_DWE/autoencoder_epoch_'+str(epoch+1)+'.pth')
        if early_stopping:
            early_stop(valid_loss,model)
            if early_stop.early_stop:
                print("Early stopping")
                break 




def train_model_DWE(autoencoder,epochs=100,batch_size=100,lr=1e-3,early_stopping=True):
    train_data_DWE = Data_DWE(train_data,file='MNIST/DWE/emd_mnist_train.npy',n_samples=700000)
    valid_data_DWE = Data_DWE(train_data,file='MNIST/DWE/emd_mnist_test.npy',n_samples=200000)
    train_DWE(autoencoder,train_data_DWE,valid_data_DWE,epochs,batch_size,lr,early_stopping)
 




