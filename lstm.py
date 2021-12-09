import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt

from preprocess import get_data
from preprocess import tokenize

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
    

    """
    Embedding
    Bidirectional LSTM
    LSTM
    Dense
    Dense
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Dropout(rate = 0.8),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dropout(rate = 0.8),
        # tf.keras.layers.LSTM(embedding_dim),
        # tf.keras.layers.Dropout(rate = 0.8),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dropout(rate = 0.8),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(train_inputs, train_labels, batch_size = 100, epochs=1, validation_data=(test_inputs, test_labels), shuffle = True, callbacks = [callback], verbose=2)
    plot_graphs(history, "accuracy") 
    plot_graphs(history, "loss") 

    predict_good_input = tokenize("Great nice man. Good. Professor. Easy. Detailed. Well done!")
    predict_bad_input = tokenize('"Statistical Inference I" is a course designed to teach students about the mathematical underpinnings of statistical tests and how to apply these statistical tests to real world problems. While the official prerequisite for this class is MATH0100, it was generally agreed that a good understanding of the material in MATH0180 was necessary in order to understand the topics of this course. Some also mentioned that previous experience with statistics, including AP Statistics, was helpful for this class, with a few cautioning future students that the horror of this class would be magnified if it was their first in statistics. While only a small fraction of students enrolled in this class showed up on the last day of class, the few who didnt stop attending class unanimously gave Professor Lawrence negative reviews. Students reported that the instructors poorly-organized PowerPoint lectures contained frequent typos which made them difficult to understand. Additionally, Lawrence went through the slides containing math and examples too quickly which left many students confused. Many wished that he could have made his lectures more linear and spent more time on examples. While the professor tried to encourage participation by using iClickers, these were generally seen as ineffective, as he asked ineffective questions and people did not take the process seriously especially given the poor overall class attendance. Respondents were also dissatisfied with the feedback process, stating that graded homework assignments and exams took too long to be returned. Graded coursework consisted of 11 homework assignments, two midterm exams, and one final. Homework assignments were described as long, difficult, and not relevant to material covered in class. Respondents also abhorred the overly-strict format of the homework assignments, which required students to rewrite the problems, define variables, and write out mathematical expressions in terms of the variables. Students were allowed to bring a cheat sheet to each exam, but some complained that the exam problems were not similar to homework problems. Respondents were split about the quality of the textbook, but agreed that the book was their primary means of learning the material. Students spent an average of eight hours on this course each week, and double this amount of time prior to exams. While many were glad that the class was finally over, most still regretted taking it. "Dont [take this class] unless you have to" was the recommendation of one of these students. Given that most people took this class because it was required, students overwhelmingly recommended trying to find a way out of the requirement.')
    print(set(test_labels))
    prediction = model.predict(predict_bad_input)
    classes = prediction.argmax(axis=-1)
    print("Good review: ", prediction)
    print("Bad review: ", model.predict(predict_bad_input))


if __name__ == '__main__':
    main()
