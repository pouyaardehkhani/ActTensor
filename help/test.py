import tensorflow as tf
import numpy as np
from act_tensor.functions import *
from act_tensor.layers import *

# functional api with classes

fmnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fmnist.load_data()

X_train = X_train /255.0
X_test = X_test / 255.0


inputs = tf.keras.layers.Input(shape=(28,28))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(128)(x)
# wanted class name
x = ReLU()(x)
output = tf.keras.layers.Dense(10,activation='softmax')(x)

model = tf.keras.models.Model(inputs = inputs,outputs=output)

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs=5)

# sequential api with classes

fmnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fmnist.load_data()

X_train = X_train /255.0
X_test = X_test / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128),
                                    # wanted class name
                                    ReLU(),
                                    tf.keras.layers.Dense(10, activation = tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs=5)
