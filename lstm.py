import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt

from preprocess import get_data

def train():
    pass


def test():
    pass


def main():
    inputs_train, inputs_test, labels_train, labels_test = get_data("jsonoutput.csv")
    model = "https://tfhub.dev/google/nnlm-en-dim50/2"
    hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)
    hub_layer(inputs_train[:3])

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.summary()

    model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])

    inputs_val = inputs_train[:300]
    partial_inputs_train = inputs_train[300:]

    labels_val = labels_train[:300]
    partial_labels_train = labels_train[300:]

    history = model.fit(partial_inputs_train,
                    partial_labels_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(inputs_val, labels_val),
                    verbose=1)

    results = model.evaluate(inputs_test, labels_test)

    print(results)


    history_dict = history.history
    history_dict.keys()

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()   # clear figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()
