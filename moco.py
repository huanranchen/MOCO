import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.utils.data.distributed
import random
import numpy as np
import itertools
import os
from tqdm import tqdm


class MoCo(nn.Module):
    def __init__(self):
        super(MoCo, self).__init__()
        torch.manual_seed(2287)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_backbone = torchvision.models.regnet_y_400mf(pretrained=False)
        self.k_backbone = torchvision.models.regnet_y_400mf(pretrained=False)
        self.transform_none = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.transform_not_none = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
        ])
        self.loss = nn.CrossEntropyLoss().to(self.device)
        self.queue = []

        self.train_loader_not_aug, self.train_loader_aug, self.queue_loader = self.get_train_loader()

        self.queue_loader = itertools.cycle(self.queue_loader)

    def q_forward(self, x):
        return self.q_backbone(x)

    def k_forward(self, x):
        return self.k_backbone(x)

    def train(self,
              t=0.1,
              lr=1e-3,
              weight_decay=0,
              batch_size=64,
              m=0.99,
              total_epoch=1000
              ):
        '''
        :param t: temperature for softmax
        :param lr:
        :param weight_decay:
        :param batch_size:
        :param m: momentum of query
        :param total_epoch:
        :return:
        '''
        if os.path.exists('q_encoder.pth'):
            self.q_backbone.load_state_dict(torch.load('q_encoder.pth'))
        if os.path.exists('k_encoder.pth'):
            self.k_backbone.load_state_dict(torch.load('k_encoder.pth'))
        for i in range(total_epoch):
            loss = self.train_one_epoch(t=t, lr=lr, weight_decay=weight_decay, batch_size=batch_size, m=m,)
            print(f'epoch {i + 1}, loss = {loss}')
            torch.save(self.q_backbone.state_dict(), 'q_encoder.pth')
            torch.save(self.k_backbone.state_dict(), 'k_encoder.pth')

    def train_one_epoch(self,
                        t=0.1,
                        lr=1e-3,
                        weight_decay=0,
                        batch_size=64,
                        m=0.99,
                        ):
        '''
        :param t: temperature for softmax
        :param lr:
        :param weight_decay:
        :param batch_size:
        :param m: momentum of query
        :return:
        '''
        optimizer = torch.optim.AdamW(self.q_backbone.parameters(), lr=lr, weight_decay=weight_decay)
        cosLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=len(self.train_loader_not_aug), eta_min=0)
        self.initialize_queue()
        epoch_loss = 0
        for (x_q, _), (x_k, _) in tqdm(zip(self.train_loader_not_aug, self.train_loader_aug)):
            optimizer.zero_grad()
            # x_q x_k both (N, D), D is the dimension of picture (C,H,W)

            q = self.q_forward(x_q.to(self.device))
            # q, k both (N, C)
            N, C = q.shape
            k = self.k_forward(x_k.to(self.device))
            k = k.detach()

            # Nx1
            l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).squeeze(2)
            # KxD
            queue = self.get_queue().detach()
            # print(queue.shape)
            # NxK
            l_neg = torch.mm(q.view(N, C), queue.permute(1, 0))

            # Nx(1+K)
            # print(l_pos.shape, l_neg.shape)
            logits = torch.cat([l_pos, l_neg], dim=1, )

            labels = torch.zeros(N, dtype=torch.long, device=self.device)
            loss = self.loss(logits / t, labels)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            cosLR.step()

            self.momentum_synchronize_backbone(m=m)
            self.enqueue(1)
            self.dequeue(1)

        epoch_loss /= len(self.train_loader_not_aug)
        return epoch_loss

    def get_train_loader(self, batch_size=64, seed=2287):
        train_set = datasets.CIFAR10('./data', train=True, transform=self.transform_none)
        torch.manual_seed(seed=seed)
        g = torch.Generator()
        train_loader_not_aug = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g, )

        train_set = datasets.CIFAR10('./data', train=True, transform=self.transform_not_none)
        torch.manual_seed(seed=seed)
        g = torch.Generator()
        train_loader_aug = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g)

        # without seed, so the order of queue loader is not same with those above
        train_set = datasets.CIFAR10('./data', train=True, transform=self.transform_none)
        queue_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        return train_loader_not_aug, train_loader_aug, queue_loader

    def get_queue(self):
        '''
        :return: a tensor, (queue_length*batch, D)
        '''
        # print(self.queue)
        return torch.cat(self.queue, dim=0)

    def initialize_queue(self, queue_length=5, batch_size=64):
        '''
        :param queue_lenth: relative length in terms of batch_size.
        :param batch_size:
        :return:
        '''
        del self.queue
        self.queue = []
        self.enqueue(queue_length)

    def enqueue(self, k, ):
        '''
        :param k: relative length in terms of batch_size.
        :return:
        '''
        for i in range(k):
            x, _ = next(self.queue_loader)
            x = x.to(self.device)
            self.queue.append(self.k_backbone(x))

    def dequeue(self, k):
        '''
        :param k:
        :return: relative length in terms of batch_size.
        '''
        assert len(self.queue) > k, 'cant dequeue because queue are empty!!!'
        for i in range(k):
            self.queue.pop(0)

    def empty_queue(self):
        self.queue = None

    def momentum_synchronize_backbone(self, m):
        with torch.no_grad():
            for i, j in zip(self.q_backbone.parameters(), self.k_backbone.parameters()):
                j = m * j + (1 - m) * i
