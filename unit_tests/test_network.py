import unittest
from streamer_net.dataset import PictureDataset
from streamer_net.network_definition import SimpleCNN
from torchvision.transforms import ToTensor
import os


class TestDataset(unittest.TestCase):
    def test_can_forward_eval(self):
        dir = os.path.join(os.getcwd(), "unit_tests/emote_training_data")
        dataset = PictureDataset(dir)
        model = SimpleCNN(dataset.n_categories)
        image, label = dataset.__getitem__(0)
        outputs = model(image)

    def test_can_train(self):
        dir = os.path.join(os.getcwd(), "unit_tests/emote_training_data")
        dataset = PictureDataset(dir)
        model = SimpleCNN(dataset.n_categories)
