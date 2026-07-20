import numpy as np

from streamer_net.network_definition import SimpleCNN
from streamer_net.dataset import PictureDataset
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import torch
from data_augmentator import augmentation_transform
from streamer_net.training_loop import train_single_epoch
EPOCHS = 120

dir = os.path.join(os.getcwd(), "emote_training_data")
dataset = PictureDataset(dir, transform = augmentation_transform)
train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

training_loader = torch.utils.data.DataLoader(train_set, batch_size=30, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=30, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
model = SimpleCNN(dataset.n_categories)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

full_train_loss = []
full_test_loss = []

for epoch in range(EPOCHS):
    train_loss, test_loss = train_single_epoch(training_loader, test_loader, optimizer, loss_fn, model)

    full_train_loss.append(np.mean(train_loss))
    full_test_loss.append(np.mean(test_loss))

plt.plot(full_train_loss, label = "train loss")
plt.plot(full_test_loss, label = "test loss")
plt.legend()
plt.show()
torch.save(model.state_dict(), "model")