import unittest
from streamer_net.dataset import PictureDataset
from streamer_net.network_definition import SimpleCNN
from streamer_net.accuracy import accuracy
import torch
import os

class TestDataset(unittest.TestCase):
    def test_accuracy(self):
        logits = torch.Tensor([[1,2], [3,2], [10,1]])
        labels = torch.Tensor([1, 0, 1]) #accuracy should be 2/3 for this setup
        self.assertEqual(accuracy(logits, labels), 2/3)