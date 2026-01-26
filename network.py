import numpy as np

## np.random.randn(x, y)
### Create matrix n x columns and y rows with random values

## zip(list a, list b)
### add lists together: [(a[0], b[0]), (a[1], b[1])]

class Network():
    def __init__(self, network_layers, starting_values):
        self.layers_num = len(network_layers)
        self.network_layers = network_layers
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(network_layers[:-1], network_layers[1:])] 
        self.biases = [np.random.randn(y, 1) for y in network_layers[1:]] ## 
        self.current_layer = 0
        self.starting_values = starting_values
        self.current_layer_values = starting_values


    def ReLU(self, node_value): ## a is a singular node on the layer/matrix
        return np.maximum(0, node_value)

    ## Send matrix data through weigths from previous layer to new layer
    ### Input (a): Old matrix layer. Output: New Matrix layer
    def feed_forward(self):
        weighed_values = np.dot(self.weights[self.current_layer], self.current_layer_values)
        ## sum(weights*outputs) = np.dot(weights, outputs)

        new_values = []
        for weight, bias in zip(weighed_values, self.biases[self.current_layer]):
            new_values.append(self.ReLU(weight + bias[0]))
            ## relu(sum(weights*outputs) + bias) for each node

        self.current_layer += 1
        if self.current_layer == self.layers_num-1:
            self.current_layer = None
        
        self.current_layer_values = new_values
        return self.current_layer_values
    
    def pass_all_layers(self):
        while self.current_layer != None:
            self.feed_forward()
        result = self.current_layer_values
        # reset
        self.current_layer = 0
        self.current_layer_values = self.starting_values

        return result

    def cost_function(self):
        pass

    def backpropagation(self):
        pass

network = Network([10, 5, 5, 3], [np.random.rand()]*10)
print(network.pass_all_layers())
