import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn

def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    # Destroy the process group
    dist.destroy_process_group()

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def train(rank, world_size):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    
    # Create model and move it to the corresponding GPU
    model = SimpleNN().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Use a distributed sampler
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, sampler=train_sampler)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # Training loop
    ddp_model.train()
    for epoch in range(2):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Rank {rank}, Epoch {epoch}, Loss: {epoch_loss / len(train_loader)}")
    
    cleanup()

def main():
    # Number of GPUs available
    world_size = int(torch.cuda.device_count()/2) if torch.cuda.device_count()==8 else 4
    print("World size or available GPUs is {0}".format(world_size))

    if world_size > 1:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("This example requires at least 2 GPUs to run")
    """
    """


if __name__ == "__main__":
    main()
