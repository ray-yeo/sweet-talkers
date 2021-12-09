import numpy as np
import random
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model

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
    # train_inputs = np.array(train_inputs)
    # test_inputs = np.array(test_inputs)
    # train_labels = np.array(train_labels)
    # test_labels = np.array(test_labels)
    correct = 0
    total = 0
    for ind, input in enumerate(test_inputs):
        rand = random.randint(1,3)
        if (rand == test_labels[ind]):
            correct += 1
        total += 1
    
    print("Dummy Accuracy: ", correct/total)

if __name__ == '__main__':
    main()
