import unittest
from streamer_net.dataset import PictureDataset
from streamer_net.network_definition import SimpleCNN
from streamer_net.training_loop import train_single_epoch
from torchvision.transforms import ToTensor
import os
import torch


class TestDataset(unittest.TestCase):
    def test_can_forward_eval(self):
        dir = os.path.join(os.getcwd(), "unit_tests/emote_training_data")
        dataset = PictureDataset(dir)
        model = SimpleCNN(dataset.n_categories)
        image, label = dataset.__getitem__(0)
        outputs = model(image)

    def test_can_train(self):
        #The training data is just two colors, a few training steps should be sufficient to get zero training loss

        torch.manual_seed(1234)
        dir = os.path.join(os.getcwd(), "unit_tests/emote_training_data")
        dataset = PictureDataset(dir, transform=ToTensor())

        training_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        loss_fn = torch.nn.CrossEntropyLoss()
        model = SimpleCNN(dataset.n_categories)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10): #epoch set intentionally low to not slow testing too much
            #loop takes a test set, but that doesn't matter for this test
            train_loss, _ = train_single_epoch(training_loader, training_loader, optimizer, loss_fn, model)

        self.assertAlmostEqual(train_loss, 0, 5)


