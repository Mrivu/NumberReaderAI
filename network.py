import math
import data_handler as dh
import numpy as np

## np.random.randn(x, y)
### Create matrix n x columns and y rows with random values

## zip(list a, list b)
### add lists together: [(a[0], b[0]), (a[1], b[1])]

class Network():
    """
    Network class handels network init, 
    forward propagation, backward propagation and gradient descent; aka. training the network

    Params:
    network_layers: list of network layers and node count per layer. 
    First entry is input layer and last entry is output layer.

    weights: None at default. If None generate random weights. 
    If existing weights are passed, generate network with those weights.

    biases: None at default. If None generate random biases. 
    If existing biases are passed, generate network with those biases.
    """

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

    # Not in use
    def ReLU(self, node_value): ## a is a singular node on the layer/matrix
        return np.maximum(0, node_value)

    def sigmoid(self, node_value):
        """
        Clamp value between -1 and 1
        """

        return 1/(1 + math.e**-node_value)

    def derived_sigmoid(self, node_value):
        return self.sigmoid(node_value)*(1 - self.sigmoid(node_value))

    def feed_forward(self):
        """
        Send image data through weigths from previous layer to new layer.
        """

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
        """
        Pass image data through all layers.
        Output Output layer values.

        Params:
        starting_values: image data
        """

        self.current_layer_values = starting_values
        while self.current_layer is not None:
            self.feed_forward()
        result = self.current_layer_values
        # reset
        self.current_layer = 0
        self.current_layer_values = None

        result = self.softmax(result)
        return result

    def cost_function(self, correct_number, values):
        """
        Get error of prediction; aka. how wrong the prediction was.
        
        Parasms:
        correct_number: Target output of network.
        values: Actual output list of network.
        """

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
        """
        Turn output list into propability percentages.
        """

        return np.exp(output) / np.sum(np.exp(output))

    def test_network(self, image, label):
        """
        Test network on single image

        Params:
        image: Image data
        label: Correct prediction
        """

        feed_forward_output = self.pass_all_layers(dh.grayscale_to_sigmoid(image))
        prediction = np.argmax(feed_forward_output)
        if prediction == label:
            return True
        return False
    
    def gradient_descent(self, epochs):
        """
        Train neural network. 
        Run forward propagation on image and run backpropagation to adjust weights and biases.

        Run on each image in train database param(epoch) times.
        """

        accuracy = 0
        print("Traning Algorithim with " + str(epochs) + " epochs... ")
        for e in range(epochs):
            train_images, train_labels = dh.get_shuffled_training_data()
            size = len(train_images)
            for i in range(size):
                train_image = train_images[i]
                train_label = train_labels[i]
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
                if i % (size / 10) == 0:
                    print("Epoch " + str(e+1) +
                          " - Training progress: " + str((i/size) * 100 ) + "%")
                    print("Epoch " + str(e+1) + " - Train accuracy: " + str(accuracy / (size / 10)))
                    accuracy = 0

    def backpropagation(self, correct_number, values):
        """
        Find how much weights and biases need to be adjusted to get closer to the target answer.

        Params:
        correct_number: Target answer
        values: Actual network output
        """

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
