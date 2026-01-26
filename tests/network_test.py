import unittest
import network as net
import random
import numpy as np

class NetworkTest(unittest.TestCase):
    def setUp(self):
        self.network_A = net.Network([10, 5, 5, 3], [np.random.rand()]*10)
        self.network_B = net.Network([8, 4, 4, 2], [np.random.rand()]*8)

    def get_random_network(self):
        random_number = random.randint(2, 10)
        return random_number, net.Network([random_number*3, random_number*2, random_number*3, random_number], [np.random.rand()]*random_number*3)

    def test_correct_output_size(self):
        self.assertEqual(len(self.network_A.pass_all_layers()), self.network_A.network_layers[-1])
        self.assertEqual(len(self.network_B.pass_all_layers()), self.network_B.network_layers[-1])

        random_number, network_C = self.get_random_network()
        self.assertEqual(len(network_C.pass_all_layers()), random_number)

    def test_identical_output(self):
        run_1 = self.network_A.pass_all_layers()
        run_2 = self.network_A.pass_all_layers()
        self.assertEqual(run_1, run_2)

        run_1 = self.network_B.pass_all_layers()
        run_2 = self.network_B.pass_all_layers()
        self.assertEqual(run_1, run_2)

        random_number, network_C = self.get_random_network()
        run_1 = network_C.pass_all_layers()
        run_2 = network_C.pass_all_layers()
        self.assertEqual(run_1, run_2)