import unittest
import network as net
import random
import numpy as np
import data_handler as dh

class NetworkTest(unittest.TestCase):
    def setUp(self):
        pass

    def get_image_network(self):
        return net.Network([784, 16, 16, 10])
    
    def test_cost_decreasing_and_weight_change(self):
        network_C = self.get_image_network()
        random_image, random_image_label = dh.random_image_test()

        weights_one = [w.copy() for w in network_C.weights]
        first_cost = network_C.cost_function(random_image_label, network_C.pass_all_layers(random_image))
        network_C.gradient_descent(1)
        weights_two = network_C.weights
        second_cost = network_C.cost_function(random_image_label, network_C.pass_all_layers(random_image))
        self.assertGreater(first_cost, second_cost)

        changed = any(
            not np.array_equal(w1, w2)
            for w1, w2 in zip(weights_one, weights_two)
        )
        self.assertTrue(changed)
    