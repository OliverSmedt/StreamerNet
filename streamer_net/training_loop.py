import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.nn.modules.module import Module

def train_single_epoch(training_loader : DataLoader,
                       test_loader: DataLoader,
                       optimizer : Optimizer,
                       loss_fn : _Loss,
                       model : Module):
    
    train_loss = []
    test_loss = []

    for i, data in enumerate(training_loader):
        inputs, labels = data # forward eval
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels) # backpropagation
        loss.backward()

        optimizer.step() # update model

        train_loss.append(loss.item()) # save loss

    for i, data in enumerate(test_loader):
        inputs, labels = data
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        test_loss.append(loss.item())

    return np.mean(train_loss), np.mean(test_loss)


