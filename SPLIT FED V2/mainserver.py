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

#  fedavg
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

# grad avg
def model_avg(models, num_clients, n, m):
    global_model_dict = {key: torch.zeros_like(param) for key, param in models[0].state_dict().items()}

    for i, model in enumerate(models):
        for key, param in model.state_dict().items():
            global_model_dict[key] += (n[0, i] / m) * param

    return global_model_dict


#num of sample and total sample
def recv_data(N):
    n = torch.zeros(size=(1,N-2))
    for i in range(N-2):
        temp = torch.tensor([0])
        print(temp.size())
        recv(temp,src = i)
        n[0,i] = temp.item()
    m = n.sum()
    return n,m
          

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
criterion=nn.CrossEntropyLoss()
num_clients = 2

model =CNN()
optim = torch.optim.SGD(model.parameters(), lr=0.005)

# saved_model_dict = torch.load("/app/model.pth")
# filter_layer ={layer_name:params for layer_name,params in saved_model_dict.items() if not layer_name.startswith('conv1')}
# model.load_state_dict(filter_layer)
# model.eval()


def run(num_epoch, batch_size, n, m,N, 
        model=model, optim=optim, 
        send=send, recv=recv, criterion=criterion):
    
    i = 0

    
    for epoch in range(num_epoch):
        for batch in range(int(m.item()) // batch_size):
            smashed_data = torch.zeros(size=(batch_size, 10, 12, 12), requires_grad=True)
            recv(smashed_data, src=i)

            target = torch.zeros(size=(batch_size,),dtype = torch.long)
            recv(target, src=i)
            print(f"Server: Received target shape {target.shape}, values: {target[:5]}")

            target = target.view(-1).to(dtype=torch.long)  
            
 
            logits = model(smashed_data)
            loss = criterion(logits, target)
            print(f"Logits shape: {logits.shape}, Target shape: {target.shape}")
            loss.backward()  
            gradient = smashed_data.grad.clone()
            send(gradient, dst=i)

            optim.step()
            optim.zero_grad()

            i = (i + 1) % num_clients

            print(f"Epoch {epoch}, Batch {batch}, Loss: {loss.item()}")

                


world_size = 4
rank = 2
num_epoch = 5
batch_size = 250
round = 5
init(rank=rank,world_size=world_size)
n,m = recv_data(N=world_size)
print(m)
run(num_epoch=num_epoch,n=n,m=m,N=world_size,batch_size=batch_size)

terminate(rank=rank)
