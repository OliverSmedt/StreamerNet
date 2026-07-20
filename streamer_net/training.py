import numpy as np

from streamer_net.network_definition import SimpleCNN
from streamer_net.dataset import PictureDataset
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import torch
from data_augmentator import augmentation_transform
EPOCHS = 120

dir = os.path.join(os.getcwd(), "emote_training_data")
dataset = PictureDataset(dir, transform = augmentation_transform)
train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

training_loader = torch.utils.data.DataLoader(train_set, batch_size=30, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=30, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
model = SimpleCNN(dataset.n_categories)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



full_train_loss = []
full_test_loss = []
for epoch in range(EPOCHS):
    train_loss = []
    test_loss = []
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        train_loss.append(loss.item())

    for i, data in enumerate(test_loader):
        inputs, labels = data
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        test_loss.append(loss.item())

    full_train_loss.append(np.mean(train_loss))
    full_test_loss.append(np.mean(test_loss))



plt.plot(full_train_loss, label = "train loss")
plt.plot(full_test_loss, label = "test loss")
plt.legend()
plt.show()
torch.save(model.state_dict(), "model")