import torch
import torch.nn as nn
import torch.nn.functional as F # Often contains activation functions and other utilities
NUM_CATEGORIES = sum(1 for line in open('emote_training_data/categories.txt'))



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Layer definitions
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        # Max Pooling Layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        # Max Pooling Layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        # Fully connected layers
        # The input features for fc1 depends on the output shape after pooling
        self.fc1 = nn.Linear(in_features=32 * 11 * 9, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=NUM_CATEGORIES) # Output for n classes

    def forward(self, x):
        #conv layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # Flatten
        x = x.view(-1, 32 * 11 * 9)

        # Apply FC1 and ReLU
        x = F.relu(self.fc1(x))

        # Apply FC2 (output layer, no activation here, typically applied with loss function)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    net = SimpleCNN()
    model = SimpleCNN()
    print(model)

    # Create a dummy input tensor (batch of 4 images, 1 channel, 28x28)
    # Requires gradient tracking if you intend to train
    dummy_input = torch.randn(4, 3, 176, 144)

    # Pass the input through the model (forward pass)
    output = model(dummy_input)
    print(output)