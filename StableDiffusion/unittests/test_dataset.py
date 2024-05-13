import unittest
from collections import OrderedDict

from StableDiffusion.dataset import CelebADataset


class TestCelebADataset(unittest.TestCase):
    def setUp(self):
        self.dataset = CelebADataset(is_train=True)

    def test_len(self):
        self.assertIsInstance(len(self.dataset), int)

    def test_load_images(self):
        images, labels, masks = self.dataset.load_images(self.dataset.dataset_root)
        self.assertIsInstance(images, list)
        self.assertIsInstance(labels, list)
        self.assertIsInstance(masks, list)

    def test_get_item(self):
        item = self.dataset.__getitem__(0)
        self.assertIsInstance(item, OrderedDict)
        self.assertTrue('img' in item)
        self.assertTrue('text' in item)
        self.assertTrue('mask' in item)


if __name__ == '__main__':
    unittest.main()
