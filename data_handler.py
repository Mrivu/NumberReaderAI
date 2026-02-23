from mnist import MNIST
import random
import numpy as np

mndata = MNIST('datasets')

training_images, training_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

def get_training_data():
    return training_images, training_labels

def get_shuffled_training_data():
    shuffled_images = []
    shuffled_labels = []
    index_shuf = list(range(len(training_images)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        shuffled_images.append(training_images[i])
        shuffled_labels.append(training_labels[i])
    return shuffled_images, shuffled_labels

def get_test_data():
    return test_images, test_labels

def random_image_test():
    index = random.randrange(0, len(test_images))
    return test_images[index], test_labels[index]

def random_image_train():
    index = random.randrange(0, len(training_images))
    return training_images[index], training_labels[index]

def view(image, label):
    return print(mndata.display(image), "Number : " + str(label))

def show_image_data(image):
    return print(image)

def grayscale_to_sigmoid(image):
    image = np.array(image, dtype=np.float32)
    image = image / 127.5 - 1.0
    return image