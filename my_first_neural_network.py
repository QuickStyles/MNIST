from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers

# keras includes the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

network.load_weights('./checkpoints/my_first_neural_network')
