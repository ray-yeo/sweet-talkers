import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt

from preprocess import get_data

vocab_size = 5000
embedding_dim = 16
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

    # print(set(test_labels))
    # print(set(train_labels))

    # print("1: ", test_labels.count(1)) # 0
    # print("2: ", test_labels.count(2)) # 1
    # print("3: ", test_labels.count(3)) # 21
    # print("(test) 4: ", test_labels.count(4)) # 269
    # print("(train) 4: ", train_labels.count(4)) # 269
    # print("5: ", test_labels.count(5)) # 130

    train_inputs = np.array(train_inputs)
    test_inputs = np.array(test_inputs)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)


    print('shapes')
    print(train_inputs.shape)
    print(test_inputs.shape)
    print(train_labels.shape)
    print(test_labels.shape)



    

    callback = tf.keras.callbacks.EarlyStopping(monitor = "loss", patience = 3, restore_best_weights = True)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Dropout(rate = 0.5),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dropout(rate = 0.5),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dropout(rate = 0.5),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(train_inputs, train_labels, batch_size = 100, epochs=10, validation_data=(test_inputs, test_labels), shuffle = True, callbacks = [callback], verbose=2)
    plot_graphs(history, "accuracy") 
    plot_graphs(history, "loss") 

    # print(model.predict("Class bad teacher sucks balls. Do not recommend. Bad. Long. Duration. Slow. Difficult. Don't like him."))

if __name__ == '__main__':
    main()
