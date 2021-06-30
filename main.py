import argparse
import time
import torch
import torchvision
from torch import distributed as dist
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.functional as F


class LinModel(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(LinModel, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.linear(x)
        out = F.softmax(out, dim=-1)
        return out


def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, help="local gpu id")
args = parser.parse_args()

batch_size = 128
epochs = 5
lr = 0.001

dist.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(args.local_rank)
global_rank = dist.get_rank()

print(f'global_rank = {global_rank} local_rank = {args.local_rank}')

net = LinModel(10, 2)
net.cuda()
net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank)
dataset = [torch.randn(10) for i in range(10000)]
labels = [torch.randint(2, [1]).item() for i in range(10000)]
trainset = list(zip(dataset, labels))
sampler = DistributedSampler(trainset)
train_loader = DataLoader(trainset,
                          batch_size=batch_size,
                          shuffle=False,
                          pin_memory=True,
                          sampler=sampler)
criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=lr)

net.train()
world_size = 4
for e in range(epochs):
    sampler.set_epoch(e)
    for idx, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.cuda()
        labels = labels.cuda()
        output = net(imgs)
        loss = criterion(output, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        reduce_loss(loss, global_rank, world_size)
        if idx % 10 == 0 and global_rank == 0:
            print('Epoch: {} step: {} loss: {}'.format(e, idx, loss.item()))
# net.eval()
# with torch.no_grad():
#     cnt = 0
#     total = len(val_loader.dataset)
#     for imgs, labels in val_loader:
#         imgs, labels = imgs.cuda(), labels.cuda()
#         output = net(imgs)
#         predict = torch.argmax(output, dim=1)
#         cnt += (predict == labels).sum().item()

# if global_rank == 0:
#     print('eval accuracy: {}'.format(cnt / total))