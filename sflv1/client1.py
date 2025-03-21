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


#loading dataset
class TrainDataset(Dataset):
    def __init__(self,transform=None,path_x='/app/X_train.npy',path_y='/app/y_train.npy'):
        self.x= np.load(path_x)
        self.y = np.load(path_y)
        self.n = self.x.shape[0]
        self.height = self.x.shape[1]
        self.width = self.x.shape[2]
        self.transform =transform
        
    def __getitem__(self, index):
        sample = self.x[index],self.y[index]
        
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.n
    
    def shape(self):
        return self.n,self.height,self.width
    

#transform
class ToTensor:
    def __call__(self,sample):
        input,target=sample
        input = input.reshape((1,28,28)).astype('float32')
        return torch.from_numpy(input),target
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        return x

#batch and loss
batch_size = 250
transform = ToTensor()
dataset = TrainDataset(path_x='/splfed/x_fed_1.npy',path_y='/splfed/y_fed_1.npy',transform=transform)
dataloader = DataLoader(dataset=dataset,shuffle=False,batch_size=batch_size)

model = CNN()
# criterion=nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.005)

#training loop
def run(num_epoch, main_server,fed_server, model=model, optim=optim,
        dataloader=dataloader):

    
    for epoch in range(num_epoch):
        recv_model(model=model,src=fed_server)
        for i, (feature,target) in enumerate(dataloader):
            
            smashed_data = model(feature)
            send(smashed_data,dst=main_server)
            
            target = target.to(dtype=torch.long)  # Flatten & Convert to LongTensor
            send(target, dst=main_server)
            print(f"Client {rank}: Sending target {target.shape}, values: {target[:5]}")
            
            gradient = torch.zeros_like(smashed_data)
            recv(gradient,src=main_server)
            
            smashed_data.backward(gradient)
            optim.step()
            
            optim.zero_grad()
            
            print(f" epoch : {epoch}/{num_epoch} {i}/{len(dataloader)} ")
    
        send_model(model=model,dst=fed_server)
        
    recv_model(model=model,src=fed_server)

world_size = 4
rank = 0
main_server = 2
fed_server = 3
num_epoch = 5
batch_size = 250
n=torch.tensor([len(dataset)])
print(n.size())
init(rank=rank,world_size=world_size)
send(arr=n,dst=main_server)
send(arr=n,dst=fed_server)

run(num_epoch,main_server=main_server,fed_server=fed_server)

terminate(rank=rank)


    