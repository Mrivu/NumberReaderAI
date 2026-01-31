from network import Network
import data_handler as data_handler
import os
import numpy as np

class Interface():
    def __init__(self):
        self.network = None
        self.image = None
        self.label = None

    def display_interface(self):
        os.system('cls||clear')
        print("="*50)
        print(" <NUMBER READER AI> By: Iivari van Uden")
        print("="*50)
        print("COMMANDS: ")
        print("- generate (generate new network)")
        print("- ff (feed forward - send image through the network or random data if image not loaded)")
        print("- load (load random image from database)")
        print("- view (view loaded image from database)")
        print("- remove (remove loaded image from database)")
        print("- Read README for more detailed instructions")
        return input("- Enter Command > ")
    
    def handle_commands(self, command):
        print()
        match str.lower(command):
            case "generate":
                print("Generating network with neuron layers: 784, 16, 16, 10...")
                print("Initializing random weights and biases...")
                self.network = Network([784, 16, 16, 10])
                print("Network generated")
                return input("- Press enter to continue > ")
            case "ff":
                if self.network is None:
                    print("Error: No network generated")
                    return input("- Press enter to continue > ")
                if self.image is None:
                    answer = self.network.pass_all_layers([np.random.rand()]*784)
                    result = []
                    for i in answer:
                        result.append(float(i))
                    print(result)
                else:
                    answer = self.network.pass_all_layers(self.image)
                    result = []
                    for i in answer:
                        result.append(float(i))
                    print(result)
                    print("Cost: " + str(self.network.cost_function(self.label, answer)))
                    print("Probabilities: " + str(self.network.softmax(answer)))
                return input("- Press enter to continue > ")
            case "load":
                image, label = data_handler.random_image()
                print("Got image, (" + str(label) + ")")
                self.image = image
                self.label = label
                return input("- Press enter to continue > ")
            case "view":
                if self.image and self.label:
                    data_handler.view(self.image, self.label)
                else:
                    print("Error: No image loaded")
                return input("- Press enter to continue > ")
            case "remove":
                self.image = None
                self.label = None
                print("Loaded image removed")
                return input("- Press enter to continue > ")
            case _:
                print("Error: Command not recognized")
                return input("- Press enter to continue > ")


interface = Interface()

while True:
    interface.handle_commands(interface.display_interface())
