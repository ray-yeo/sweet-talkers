import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt

from preprocess import get_data

vocab_size = 1000
embedding_dim = 64
max_length = 437
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

def train():
    pass


def test():
    pass


def main():
    train_inputs, test_inputs, train_labels, test_labels = get_data("jsonoutput.csv")
    
    model = tf.keras.Sequential([
        # Add an Embedding layer expecting input vocab of size 5000, and output embedding di
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        # use ReLU in place of tanh function since they are very good alternatives of each o
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        # Add a Dense layer with 6 units and softmax activation.
        # When we have multiple outputs, softmax convert outputs layers into a probability d
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    num_epochs = 10
    history = model.fit(train_inputs, train_labels, epochs=num_epochs, validation_data=(test_inputs, test_labels), verbose=2)

if __name__ == '__main__':
    main()
