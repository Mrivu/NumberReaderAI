import unittest
import network as net
import random
import numpy as np

class NetworkTest(unittest.TestCase):
    def setUp(self):
        self.network_A = net.Network([10, 5, 5, 3])
        self.network_B = net.Network([8, 4, 4, 2])

    def get_random_network(self):
        random_number = random.randint(2, 10)
        return random_number, net.Network([random_number*3, random_number*2, random_number*3, random_number])

    def test_correct_output_size(self):
        self.assertEqual(len(self.network_A.pass_all_layers([np.random.rand()]*10)), self.network_A.network_layers[-1])
        self.assertEqual(len(self.network_B.pass_all_layers([np.random.rand()]*8)), self.network_B.network_layers[-1])

        random_number, network_C = self.get_random_network()
        self.assertEqual(len(network_C.pass_all_layers([np.random.rand()]*random_number*3)), random_number)

    def test_identical_output(self):
        random_values = [np.random.rand()]*10
        run_1 = self.network_A.pass_all_layers(random_values)
        run_2 = self.network_A.pass_all_layers(random_values)
        self.assertEqual(run_1, run_2)

        random_values = [np.random.rand()]*8
        run_1 = self.network_B.pass_all_layers(random_values)
        run_2 = self.network_B.pass_all_layers(random_values)
        self.assertEqual(run_1, run_2)

        random_number, network_C = self.get_random_network()
        random_values = [np.random.rand()]*random_number*3
        run_1 = network_C.pass_all_layers(random_values)
        run_2 = network_C.pass_all_layers(random_values)
        self.assertEqual(run_1, run_2)