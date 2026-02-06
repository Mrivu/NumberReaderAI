import numpy as np
import math
import random
import data_handler as dh

## np.random.randn(x, y)
### Create matrix n x columns and y rows with random values

## zip(list a, list b)
### add lists together: [(a[0], b[0]), (a[1], b[1])]

class Network():
    def __init__(self, network_layers, weights=None, biases=None):
        self.layers_num = len(network_layers)
        self.network_layers = network_layers
        if not weights:
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(network_layers[:-1], network_layers[1:])]
        else:
            self.weights = weights
        if not biases: 
            self.biases = [np.random.randn(y, 1) for y in network_layers[1:]]
        else:
            self.biases = biases
        self.current_layer = 0
        self.current_layer_values = None
        self.previous_values = []


    def ReLU(self, node_value): ## a is a singular node on the layer/matrix
        return np.maximum(0, node_value)
    
    def sigmoid(self, node_value):
        return 1/(1 + math.e**-node_value)
    
    def derived_sigmoid(self, node_value):
        return self.sigmoid(node_value)*(1 - self.sigmoid(node_value))

    ## Send matrix data through weigths from previous layer to new layer
    ### Input (a): Old matrix layer. Output: New Matrix layer
    def feed_forward(self):
        weighed_values = np.dot(self.weights[self.current_layer], self.current_layer_values)
        ## sum(weights*outputs) = np.dot(weights, outputs)

        new_values = []
        for weight, bias in zip(weighed_values, self.biases[self.current_layer]):
            new_values.append(self.sigmoid(weight + bias[0]))
            ## relu(sum(weights*outputs) + bias) for each node

        self.current_layer += 1
        if self.current_layer == self.layers_num-1:
            self.current_layer = None
        
        self.previous_values.append(self.current_layer_values)
        self.current_layer_values = new_values

        return self.current_layer_values
    
    def pass_all_layers(self, starting_values):
        self.current_layer_values = starting_values
        while self.current_layer != None:
            self.feed_forward()
        result = self.current_layer_values
        # reset
        self.current_layer = 0
        self.current_layer_values = None
        #self.previous_values = []

        result = self.softmax(result)
        return result
    
    def cost_function(self, correct_number, values):
        correct_answer = [0.0]*self.network_layers[-1]
        correct_answer[correct_number] = 1.0
        cost = 0
        for value, answer in zip(values, correct_answer):
            cost += (value - answer)**2
        return cost
    
    def derived_cost_function(self, correct_number, values):
        correct_answer = [0.0]*self.network_layers[-1]
        correct_answer[correct_number] = 1.0
        result = []
        for value, answer in zip(values, correct_answer):
            result.append(2*(value - answer))
        return np.array(result)
    
    def softmax(self, output):
        return np.exp(output) / np.sum(np.exp(output))
    
    def test_network(self, batch_size):
        accuracy = 0.0
        ## Fix grayscale value input / 255
        for i in range(batch_size):
            test_image, test_label = dh.random_image_test()
            feed_forward_output = self.pass_all_layers(dh.grayscale_to_sigmoid(test_image))
            prediction = np.argmax(feed_forward_output)
            if prediction == test_label:
                accuracy += 1.0
        return accuracy / batch_size


    def gradient_descent(self, batch_size):
        accuracy = 0
        for i in range(batch_size):
            train_image, train_label = dh.random_image_train()
            feed_forward_output = self.pass_all_layers(dh.grayscale_to_sigmoid(train_image))

            #cost = self.cost_function(train_label, feed_forward_output)
            weight_change, bias_change = self.backpropagation(train_label, feed_forward_output)

            prediction = np.argmax(feed_forward_output)
            if prediction == train_label:
                accuracy += 1.0

            learning_rate = 0.01

            for layer_num in range(len(self.weights)):
                self.weights[layer_num] -= learning_rate * weight_change[layer_num]
                for b in range(len(self.biases[layer_num])):
                    self.biases[layer_num][b] -= learning_rate * bias_change[layer_num][b]
            if i % 100 == 0:
                print("Iter: " + str(i))
                print("Train accuracy: " + str(accuracy / 100))
                accuracy = 0

    def backpropagation(self, correct_number, values):
        weight_change = []
        bias_change = []

        ## Output Layer
        c_al = self.derived_cost_function(correct_number, values)

        weighed_values = np.dot(self.weights[-1], self.previous_values[-1])

        new_values = []
        for weight, bias in zip(weighed_values, self.biases[-1]):
            new_values.append(self.derived_sigmoid(weight + bias[0]))

        al_zl = new_values
        zl_wl = self.previous_values[-1]

        #onehot = [0]*10
        #onehot[correct_number] = 1

        delta = c_al * al_zl

        weight_change.append(np.outer(delta, zl_wl))
        bias_change.append(delta)

        ## Hidden Layers
        for layer_num in range(1, self.layers_num-1):
            weighed_values = np.dot(self.weights[-1-layer_num], self.previous_values[-1-layer_num])

            new_values = []
            for weight, bias in zip(weighed_values, self.biases[-1-layer_num]):
                new_values.append(self.derived_sigmoid(weight + bias[0]))

            wl = self.weights[-layer_num]
            new_delta = np.dot(wl.T, delta) * new_values
            delta = new_delta

            weight_change.append(np.outer(new_delta, self.previous_values[-1-layer_num]))
            bias_change.append(new_delta)

        weight_change.reverse()
        bias_change.reverse()
        return weight_change, bias_change

#network = Network([784, 16, 16, 10])
#print(answer)
#print(network.cost_function(2, answer))
#print(network.softmax(answer))
#w, b = network.backpropagation(2, answer)
#print(len(b))
#for i in b:
#    print(i)
#for i in w:
#   print(i)
#network.gradient_descent(5000)