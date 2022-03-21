import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm
from torchvision import transforms


class Classifier(nn.Module):
    def __init__(self, encoder, classes=10):
        '''
        :param encoder: 'k_encoder' or 'q_encoder' or None
        '''
        super(Classifier, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = torchvision.models.regnet_y_400mf(pretrained=False)
        if encoder is not None:
            self.net.load_state_dict(torch.load(encoder + '.pth', map_location=device))
            self.net.requires_grad_(False)
        else:
            pass
        self.fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(1000, 200),
            nn.LeakyReLU(),
            nn.Linear(200,50),
            nn.LeakyReLU(),
            nn.Linear(50, classes),
        )



    def forward(self, x):
        x = self.net(x)
        return self.fc(x)


def get_train_loader(batch_size=64):
    train_set = datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, )

    test_set = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, )
    return train_loader, test_loader


def train(batch_size=64, lr=1e-3, total_epoch=100, mode=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_train_loader(batch_size)
    model = Classifier(mode).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loss_for_draw = []
    train_acc_for_draw = []
    valid_loss_for_draw = []
    valid_acc_for_draw = []
    best_acc = 0
    best_loss = 0

    for epoch in range(total_epoch):

        train_loss = 0
        train_acc = 0
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)

            # N, 10
            pre = model(x)
            loss = criterion(pre, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predict = torch.max(pre, dim=1)
            train_acc += (torch.sum((predict == y)).item() / batch_size)

        train_acc /= len(train_loader)
        train_loss /= len(train_loader)
        train_loss_for_draw.append(train_loss)
        train_acc_for_draw.append(train_acc)

        valid_loss = 0
        valid_acc = 0
        for x, y in tqdm(test_loader):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)

            # N, 10
            pre = model(x)
            loss = criterion(pre, y)
            valid_loss += loss.item()
            loss.backward()
            _, predict = torch.max(pre, dim=1)
            valid_acc += (torch.sum((predict == y)).item() / batch_size)

        valid_acc /= len(test_loader)
        valid_loss /= len(test_loader)
        valid_loss_for_draw.append(valid_loss)
        valid_acc_for_draw.append(valid_acc)

        if valid_acc > best_acc:
            best_acc=valid_acc
            torch.save(model.state_dict(),'model.pth')

        if valid_loss<best_loss:
            best_loss=valid_loss

        print(f'epoch {epoch}, train loss = {train_loss}, train acc = {train_acc}, valid loss = {valid_loss}, valid acc = {valid_acc}')

    return valid_loss_for_draw, valid_acc_for_draw, best_acc, best_loss