import unittest
from streamer_net.dataset import PictureDataset
import os

class TestDataset(unittest.TestCase):

    def test_load_image(self):
        dir = os.path.join(os.getcwd(), "unit_tests/emote_training_data")
        dataset = PictureDataset(dir)
        sample0 = dataset.__getitem__(0)
        sample1 = dataset.__getitem__(1)
        sample2 = dataset.__getitem__(2)
        sample3 = dataset.__getitem__(3)
        sample4 = dataset.__getitem__(4)
        sample5 = dataset.__getitem__(5)

    def test_length(self):
        dir = os.path.join(os.getcwd(), "unit_tests/emote_training_data")
        dataset = PictureDataset(dir)
        self.assertEqual(dataset.__len__(), 6)

if __name__ == '__main__':
    unittest.main()