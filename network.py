import numpy as np
import math

## np.random.randn(x, y)
### Create matrix n x columns and y rows with random values

## zip(list a, list b)
### add lists together: [(a[0], b[0]), (a[1], b[1])]

class Network():
    def __init__(self, network_layers):
        self.layers_num = len(network_layers)
        self.network_layers = network_layers
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(network_layers[:-1], network_layers[1:])] 
        self.biases = [np.random.randn(y, 1) for y in network_layers[1:]] ## 
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

    def gradient_descent(self, batch_size):
        batch = batch_size
        for i in batch:
            pass

    def backpropagation(self, correct_number, values):
        c_al = self.derived_cost_function(correct_number, values)

        ## Values for derived sigmoid
        weighed_values = np.dot(self.weights[-1], self.previous_values[-1])
        ## sum(weights*outputs) = np.dot(weights, outputs)

        new_values = []
        for weight, bias in zip(weighed_values, self.biases[-1]):
            new_values.append(self.derived_sigmoid(weight + bias[0]))

        al_zl = new_values
        zl_wl = self.previous_values[-1]

        bias = c_al * al_zl

        ## Multiple layers?
        #print("c_al: " + str(c_al))
        #print("al_zl: " + str(al_zl))
        #print("zl_wl: " + str(zl_wl))
        weight_sensitivity = np.outer(bias, zl_wl)
        bias_sensitivity = bias

        return weight_sensitivity, bias_sensitivity
        

network = Network([10, 5, 5, 3])
answer = network.pass_all_layers([np.random.rand()]*10)
#print(answer)
#print(network.cost_function(2, answer))
#print(network.softmax(answer))
print(network.backpropagation(2, answer))
