import os
import numpy as np
from network import Network
import data_handler as data_handler

class Interface():
    def __init__(self):
        self.network = None
        self.image = None
        self.label = None
        self.weights = None
        self.biases = None
        self.epochs = 30

    def display_interface(self):
        os.system('cls||clear')
        print("="*50)
        print(" <NUMBER READER AI> By: Iivari van Uden")
        print("="*50)
        print("COMMANDS: ")
        print("- generate (generate new network)")
        print("- getW (get existing weigths)")
        print("- saveW (take current weights and save them)")
        print("- ff (feed forward - send image through the network "
        "or random data if image not loaded)")
        print("- train (train the neural network)")
        print("- finderror (load image the neural network guesses wrong)")
        print("- epochs (set train size)")
        print("- test (test the neural network)")
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
                if self.biases is not None:
                    print("Initializing random weights and biases...")
                else:
                    print("Loading weights and biases")
                self.network = Network([784, 16, 16, 10], self.weights, self.biases)
                self.weights = self.network.weights
                self.biases = self.network.biases
                print("Network generated")
                return input("- Press enter to continue > ")
            case "getw":
                data = np.load("trained_weights.npz", allow_pickle=True)
                self.weights = list(data["weights"])
                self.biases = list(data["biases"])
                return input("- Press enter to continue > ")
            case "savew":
                if self.weights is not None:
                    np.savez_compressed(
                        "trained_weights.npz",
                        weights=np.array(self.network.weights, dtype=object),
                        biases=np.array(self.network.biases, dtype=object),
                    )
                else:
                    print("Error: No weights initialized")
                return input("- Press enter to continue > ")
            case "train":
                if self.network is not None:
                    self.network.gradient_descent(self.epochs)
                else:
                    print("Error: No network generated")
                return input("- Press enter to continue > ")
            case "epochs":
                new_amount = input("- Enter number of epochs "
                "(How many times the entire database is trained on) > ")
                if new_amount.isnumeric():
                    self.epochs = int(new_amount)
                else:
                    print("Error: Please enter a number!")
                return input("- Press enter to continue > ")
            case "test":
                if self.network is not None:
                    accuracy = 0.0
                    images, labels = data_handler.get_test_data()
                    size = len(images)
                    print("Testing data...")
                    for i in range(size):
                        if self.network.test_network(images[i], labels[i]):
                            accuracy += 1.0
                    print(accuracy / size)
                else:
                    print("Error: No network generated")
                return input("- Press enter to continue > ")
            case "finderror":
                if self.network is not None:
                    images, labels = data_handler.get_shuffled_test_data()
                    size = len(images)
                    error_image = None
                    error_label = None
                    print("Finding errors...")
                    for i in range(size):
                        if not self.network.test_network(images[i], labels[i]):
                            error_image = images[i]
                            error_label = labels[i]
                    self.image = error_image
                    self.label = error_label
                    print("Error image loaded")
                else:
                    print("Error: No network generated")
            case "ff":
                if self.network is None:
                    print("Error: No network generated")
                    return input("- Press enter to continue > ")
                if self.image is not None:
                    answer = self.network.pass_all_layers(
                        data_handler.grayscale_to_sigmoid(self.image))
                    prediction = np.argmax(answer)
                    print("Prediction: " + str(prediction))
                    print("Correct Label: " + str(self.label))
                    return input("- Press enter to continue > ")
                else:
                    print("No image loaded")
                return input("- Press enter to continue > ")
            case "load":
                image, label = data_handler.random_image_test()
                print("Got image, (" + str(label) + ")")
                self.image = image
                self.label = label
                return input("- Press enter to continue > ")
            case "view":
                if self.image is not None and self.label is not None:
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
