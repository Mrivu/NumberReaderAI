import unittest
import data_handler as dh

class DataTest(unittest.TestCase):
    def setUp(self):
        self.test_images, self.test_labels = dh.get_test_data()

    def test_got_data(self):
        self.assertGreater(len(self.test_images), 0)
        self.assertGreater(len(self.test_labels), 0)