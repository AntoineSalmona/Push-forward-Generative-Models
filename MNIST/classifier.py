import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from tqdm import tqdm
import torch.nn.functional as F    
from torch.optim import Adam


kwargs = {'num_workers': 1, 'pin_memory': True} 



class MnistClassifier(nn.Module):

    def __init__(self):
        super().__init__() 
        self.fc_1 = nn.Linear(784,1024)
        self.fc_2 = nn.Linear(1024,50)
        self.fc_3 = nn.Linear(50,10)

    def forward(self,x,output='end'):
        x = torch.flatten(x,start_dim=1)
        out1 = self.fc_1(x) 
        out2 = self.fc_2(F.relu(out1))
        out3 = self.fc_3(F.relu(out2))
        if output=='2':
            return out2
        elif output=='1':
            return out1
        else:
            return out3


def train_classifier(model,train_dataset,epochs=100,batch_size=128,lr=1e-4):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    optimizer = Adam(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    loop = tqdm(range(epochs))
    for epoch in loop:
        train_loss = 0
        for batch_idx, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            c = model(x)
            loss = criterion(c,y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        loop.set_postfix({'loss': train_loss/len(train_dataset)})
    
        


def test_classifier(model,test_dataset,batch_size=128):
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=True, **kwargs)
    model.eval()
    right_label = 0
    tested_label = 0
    with torch.no_grad():
        for batch_idx, (x,y) in enumerate(test_loader):
            c = torch.argmax(torch.softmax(model(x),dim=1),dim=1)
            right_label += torch.sum(y == c).item()
            tested_label += len(c)
    return right_label/tested_label