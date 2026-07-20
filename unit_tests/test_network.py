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
        #The training data is just two colors, a single big training step should be sufficient to get zero test loss
        dir = os.path.join(os.getcwd(), "unit_tests/emote_training_data")
        dataset = PictureDataset(dir, transform=ToTensor())
        train_set, test_set = torch.utils.data.random_split(dataset, [5/6, 1/6])

        training_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
        loss_fn = torch.nn.CrossEntropyLoss()
        model = SimpleCNN(dataset.n_categories)
        optimizer = torch.optim.Adam(model.parameters(), lr=1)

        train_loss, test_loss = train_single_epoch(training_loader, test_loader, optimizer, loss_fn, model)

        self.assertEqual(test_loss, 0)


