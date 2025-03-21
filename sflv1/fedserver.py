import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.distributed as dist
from datetime import timedelta
import torch.nn.functional as F
import numpy as np
import os
import logging
import random

logging.basicConfig(filename='/app/smashed.log', level=logging.INFO,format='%(asctime)s %(message)s')

# random seed settings
random.seed(43)
np.random.seed(43)
torch.manual_seed(43)

#connection
def init(rank, world_size, backend='gloo'):
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    os.environ['MASTER_ADDR'] = 'client1'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=180),
    )

    print(f"Rank {rank}: is_initialized and ready to communicate...")
    return dist.is_initialized()

def recv(arr,src):
    dist.recv(tensor=arr, src=src)

def send(arr,dst):
    dist.send(tensor=arr, dst=dst)

def terminate(rank):
    dist.destroy_process_group()
    print(f"{rank} succesfully terminated...")
    
def send_model(model, dst):
    for key, param in model.state_dict().items():
        dist.send(param, dst=dst)
        print(f"Sent {key}")
    
def recv_model(model, src):
    for key, param in model.state_dict().items():
        temp_tensor = torch.empty_like(param)  
        dist.recv(temp_tensor, src=src)  
        param.data.copy_(temp_tensor)  
        print(f"Received {key}, shape: {temp_tensor.shape}")
        
def recv_model_update(model, src, n, m):
    for key, param in model.state_dict().items():
        temp_tensor = torch.empty_like(param)  
        dist.recv(temp_tensor, src=src)
        avg_tensor = torch.zeros_like(param)
        if src == 0:
            avg_tensor = (temp_tensor * (n[0,src] / m))  
        else:
            avg_tensor = param.clone()
            avg_tensor += (temp_tensor * (n[0,src] / m)) 
        
        param.data.copy_(avg_tensor.clone())  
        print(f"Received {key}, shape: {avg_tensor.shape}")



# #loading dataset
# class TrainDataset(Dataset):
#     def __init__(self,transform=None,path_x='/app/X_train.npy',path_y='/app/y_train.npy'):
#         self.x= np.load(path_x)
#         self.y = np.load(path_y)
#         self.n = self.x.shape[0]
#         self.height = self.x.shape[1]
#         self.width = self.x.shape[2]
#         self.transform =transform
        
#     def __getitem__(self, index):
#         sample = self.x[index],self.y[index]
        
#         if self.transform:
#             sample = self.transform(sample)
#         return sample
    
#     def __len__(self):
#         return self.n
    
#     def shape(self):
#         return self.n,self.height,self.width
    

# #transform
# class ToTensor:
#     def __call__(self,sample):
#         input,target=sample
#         input = input.reshape((1,28,28)).astype('float32')
#         return torch.from_numpy(input),target

#model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        return x

#batch and loss
# batch_size = 250
# transform = ToTensor()
# dataset = TrainDataset(path_x='/app/X_train.npy',path_y='/app/y_train.npy',transform=transform)
# dataloader = DataLoader(dataset=dataset,shuffle=False,batch_size=batch_size)

model = CNN()
# criterion=nn.CrossEntropyLoss()
# optim = torch.optim.SGD(model.parameters(), lr=0.005)

def recv_data(N):
    n = torch.zeros(size=(1,N-2))
    for i in range(N-2):
        temp = torch.tensor([0])
        print(temp.size())
        recv(temp,src = i)
        n[0,i] = temp.item()
    m = n.sum()
    return n,m
    
    

#training loop
def run(num_epoch,round,n,m,N, model=model):
    for i in range(N-2):
        send_model(model=model,dst=i)
        
    for i in range(N-2):
        recv_model_update(model=model,src=i,n=n,m=m)



world_size = 4
rank = 3
num_epoch = 5
batch_size = 250
epoch = 5
init(rank=rank,world_size=world_size)
n,m = recv_data(N=world_size)
for i in range(num_epoch):
    run(num_epoch=num_epoch,round=round,n=n,m=m,N=world_size)
for i in range(world_size-2):
    send_model(model=model,dst=i)
terminate(rank=rank)