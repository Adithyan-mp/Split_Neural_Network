import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.distributed as dist
from datetime import timedelta
import torch.nn.functional as F
import numpy as np
import os
import random
import time
import logging

random.seed(43)
np.random.seed(43)
torch.manual_seed(43)

file_name = "/app/test.log"
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

def terminate(rank):
    dist.destroy_process_group()
    print(f"{rank} succesfully terminated...")
    
def send_model(model, dst):
    for key, param in model.state_dict().items():
        dist.send(param, dst=dst)
        print(f"Sent {key}")
    

def recv_model(model, src):
    for key,param in model.state_dict().items():
        dist.recv(param.data,src=src)
        print(f"recived {key}")
    
    
class TrainDataset_x(Dataset):
    def __init__(self,transform=None,path='/app/x_train_1.npy'):
        self.x= np.load(path)
        self.n = self.x.shape[0]
        self.height = self.x.shape[1]
        self.width = self.x.shape[2]
        self.transform =transform
        
    def __getitem__(self, index):
        sample_temp = self.x[index]
        
        if self.transform:
            sample_temp = self.transform(sample_temp)
            
        sample = sample_temp,index
        return sample
    
    def __len__(self):
        return self.n
    
    def shape(self):
        return self.n,self.height,self.width
    

class ToTensor:
    def __call__(self,input):
        input = input.reshape((-1,28,28)).astype('float32')
        return torch.from_numpy(input)
            
class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        return x
    
def run(num_epoch,send=send,recv=recv):
    batch_size = 250
    transform = ToTensor()
    dataset = TrainDataset_x(transform=transform,path='/app/x_train_1.npy')
    #120x250 batch size
    dataloader = DataLoader(dataset=dataset,shuffle=False,batch_size=batch_size)

    model = CNN_1()
    # optim = torch.optim.Adam(params=model.parameters(),lr=0.005)
    saved_model_state = torch.load('/app/model.pth')
    model.conv1.load_state_dict({"weight":saved_model_state['conv1.weight'],
                                "bias":saved_model_state["conv1.bias"]})
    model.eval()

    for epoch in range(num_epoch):
        for i, (data, index) in enumerate(dataloader):
            if i>0 or epoch >0 :
                recv_model(model=model,src=1)
                
            optim = torch.optim.SGD(params=model.parameters(),lr=0.005)
            output = model(data)
            
            send(output,dst=2)
            
            gradient = torch.zeros_like(output)
            recv(gradient,src=2)
            
            output.backward(gradient)
            optim.step()
            
            optim.zero_grad()
            
            
            send_model(model,dst=1)
            

        
        print(f"epoch {epoch} {i}/{len(dataloader)} completed... ")
        
    recv_model(model=model,src=1) 



num_epoch = 10
init(rank=0,world_size=3)
run(num_epoch)
terminate(rank=0)