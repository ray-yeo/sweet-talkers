import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt

from preprocess import get_data

vocab_size = 5000
embedding_dim = 64
max_length = 437

def plot_graphs(history, string): 
    plt.plot(history.history[string]) 
    plt.plot(history.history['val_'+string]) 
    plt.xlabel("Epochs") 
    plt.ylabel(string) 
    plt.legend([string, 'val_'+string]) 
    plt.show() 

def main():
    train_inputs, test_inputs, train_labels, test_labels = get_data("jsonoutput.csv", False)
    train_inputs = np.array(train_inputs)
    test_inputs = np.array(test_inputs)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)


    print(train_inputs.shape)
    print(test_inputs.shape)
    print(train_labels.shape)
    print(test_labels.shape)

    callback = tf.keras.callbacks.EarlyStopping(monitor = "loss", patience = 2, restore_best_weights = True)
    
    model = (train_inputs, train_inputs)
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_inputs, train_labels, epochs=10, validation_data=(test_inputs, test_labels), callbacks = [callback], verbose=2)
    plot_graphs(history, "accuracy") 
    plot_graphs(history, "loss") 

if __name__ == '__main__':
    main()
