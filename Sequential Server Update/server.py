import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.distributed as dist
from datetime import timedelta
import torch.nn.functional as F
import numpy as np
import os
import random

random.seed(43)
np.random.seed(43)
torch.manual_seed(43)


file_name = "/app/loss.log"
logging.basicConfig(filename=file_name,
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')
def init(rank, world_size, backend='gloo'):
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    os.environ['MASTER_ADDR'] = 'client1'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=60),
    )

    print(f"Rank {rank}: is_initialized and ready to communicate...")
    return dist.is_initialized()

def recv(arr,src):
    dist.recv(tensor=arr, src=src)

def send(arr,dst):
    dist.send(tensor=arr, dst=dst)

class TrainDataset(Dataset):
    def __init__(self,path,transform=None):
        self.x= np.load(path)
        self.n = self.x.shape[0]
        self.transform =transform
        
    def __getitem__(self, index):
        sample_temp = self.x[index]
        # print(sample_temp.shape)
        
        if self.transform:
            sample_temp = self.transform(sample_temp)
            
        sample = sample_temp
        return sample
    
    def __len__(self):
        return self.n
    
    def shape(self):
        return self.n,self.height,self.width
    
    
class ToTensor:
    def __call__(self, input):
        return torch.from_numpy(np.array(input, dtype=np.float32))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
            

batch_size = 250
transform = ToTensor()
dataset_c1 = TrainDataset(path='/app/y_train_1.npy',transform=transform)
dataloader_c1 = DataLoader(dataset=dataset_c1,shuffle=False,batch_size=batch_size)


dataset_c2 = TrainDataset(path='/app/y_train_2.npy',transform=transform)
dataloader_c2 = DataLoader(dataset=dataset_c2,shuffle=False,batch_size=batch_size)

model = CNN()
criterion=nn.CrossEntropyLoss()
optim = torch.optim.SGD(params=model.parameters(),lr=0.005)

saved_model_dict = torch.load("/app/model.pth")
filter_layer ={layer_name:params for layer_name,params in saved_model_dict.items() if not layer_name.startswith('conv1')}
model.load_state_dict(filter_layer)
model.eval()


def run(num_epoch, flag=0, batch_size=250, model=model, optim=optim, 
        dataloader_c1=dataloader_c1, dataloader_c2=dataloader_c2, 
        send=send, recv=recv, criterion=criterion):
    
    size = len(dataloader_c1)*2
    
    for epoch in range(num_epoch):
        dataiter_c1 = iter(dataloader_c1)
        dataiter_c2 = iter(dataloader_c2)

        for i in range(len(dataloader_c1)+len(dataloader_c2)):  
            
            if flag == 0:
                data_c1 = next(dataiter_c1)
                
                y_c1 = data_c1

                smashed_data = torch.zeros((batch_size, 10, 12, 12), requires_grad=True)
                recv(smashed_data, src=0)

                logits = model(smashed_data)
                log_logits = logits.detach().numpy()
                
                loss = criterion(logits, y_c1.view(-1).long())  
                
                loss.backward()
                gradient = smashed_data.grad
                
                send(gradient, dst=0)
                optim.step()
                optim.zero_grad()
                
                print(f"Epoch {epoch} [{i}/{size}] updated from rank {flag} | Loss: {loss.item():.4f}")
                
                flag = 1
            
            elif flag ==1:
                data_c2 = next(dataiter_c2)

                y_c2 = data_c2
                
                smashed_data = torch.zeros((batch_size, 10, 12, 12), requires_grad=True)
                recv(smashed_data, src=1)

                logits = model(smashed_data)
                
                loss = criterion(logits, y_c2.view(-1).long())

                loss.backward()
                gradient = smashed_data.grad

                send(gradient, dst=1)
                optim.step()
                optim.zero_grad()
                
                print(f"Epoch {epoch} [{i}/{size}] updated from rank {flag} | Loss: {loss.item():.4f}")
                
                flag = 0 
            
            if i == size-1:
                logging.info(f"loss = {loss}")


num_epoch = 10
init(rank=2,world_size=3)
run(num_epoch,batch_size=batch_size)

dist.destroy_process_group()

