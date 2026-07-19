from streamer_net.network_definition import SimpleCNN
from streamer_net.dataset import PictureDataset
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import torch

EPOCHS = 30

dir = os.path.join(os.getcwd(), "emote_training_data")
training_set = PictureDataset(dir)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=30, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
model = SimpleCNN(training_set.n_categories)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


train_loss = []
for epoch in range(EPOCHS):
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

plt.plot(train_loss)
plt.show()
torch.save(model.state_dict(), "model")