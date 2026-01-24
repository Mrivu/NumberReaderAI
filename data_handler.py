from mnist import MNIST
import random

mndata = MNIST('datasets')

training_images, training_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

def get_training_data():
    return training_images, training_labels

def get_test_data():
    return test_images, test_labels

def print_random_number():
    index = random.randrange(0, len(test_images))
    print(mndata.display(test_images[index]), test_labels[index])