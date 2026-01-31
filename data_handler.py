from mnist import MNIST
import random

mndata = MNIST('datasets')

training_images, training_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

def get_training_data():
    return training_images, training_labels

def get_test_data():
    return test_images, test_labels

def random_image():
    index = random.randrange(0, len(test_images))
    return test_images[index], test_labels[index]

def view(image, label):
    return print(mndata.display(image), "Number : " + str(label))
