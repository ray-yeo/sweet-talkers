import csv
import sklearn
import sklearn.model_selection
import math
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
STOPWORDS = set(stopwords.words('english'))

vocab_size = 1000
embedding_dim = 64
max_length = 437
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

"""
split csv file into inputs_list and labels_list
"""
def roundToNearestInt(x):
    return round(float(x))

def get_data(filename = 'jsonoutput.csv'):
    # open file and read in rows
    file = open(filename, encoding='utf-16')
    csvreader = csv.reader(file, delimiter=',')
    header = next(csvreader)
    rows = []
    for row in csvreader:
            rows.append(row)
    file.close()


    inputs_list = []
    labels_list = []

    # edition,department_code,course_num,profavg,courseavg,review_contents					
    for row in rows:
        inputs = row[5]
        labels = row[4]
        if roundToNearestInt(labels) == 0:
            continue
        for word in STOPWORDS:
            token = ' ' + word + ' '
            inputs = inputs.replace(token, ' ')
            inputs = inputs.replace(' ', ' ')
        inputs_list.append(inputs)
        labels_list.append(labels)

    
    labels_list = list(map(roundToNearestInt,labels_list))

    # Use sklearn to do train and test split 0f 0.25 | 0.75
    train_inputs, test_inputs, train_labels, test_labels = sklearn.model_selection.train_test_split(inputs_list, labels_list, test_size=0.25, random_state=0)

    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    
    tokenizer.fit_on_texts(train_inputs)
    word_index = tokenizer.word_index
    words_dict = dict(list(word_index.items())[0:10])

    train_sequences = tokenizer.texts_to_sequences(train_inputs)

    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    # max = 0
    # for seq in train_sequences:
    #     if len(seq) > max:
    #         max = len(seq)
    # print(max)

    test_sequences = tokenizer.texts_to_sequences(test_inputs)
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


    # print('Train size: ', len(train_inputs))
    # print('Test size: ', len(test_inputs))
    # print('Total:', len(train_inputs) + len(test_inputs))

    return train_padded, test_padded, train_labels, test_labels

