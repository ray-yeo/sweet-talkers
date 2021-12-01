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
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

"""
split csv file into inputs_list and labels_list
"""
def round(x):
    if (int(float(x)) >= 4):
        return 1
    else:
        return 0

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
        for word in STOPWORDS:
            token = ' ' + word + ' '
            inputs = inputs.replace(token, ' ')
            inputs = inputs.replace(' ', ' ')
        inputs_list.append(inputs)
        labels_list.append(labels)

    
    labels_list = list(map(round,labels_list))

    # Use sklearn to do train and test split 0f 0.25 | 0.75
    inputs_train, inputs_test, labels_train, labels_test = sklearn.model_selection.train_test_split(inputs_list, labels_list, test_size=0.25, random_state=0)

    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(inputs_train)
    word_index = tokenizer.word_index
    words_dict = dict(list(word_index.items())[0:10])
    print(words_dict)

    train_sequences = tokenizer.texts_to_sequences(inputs_train)
    print(train_sequences[10])
    print(inputs_list[10])


    print('Train size: ', len(inputs_train))
    print('Test size: ', len(inputs_test))
    print('Total:', len(inputs_train) + len(inputs_test))

    return inputs_train, inputs_test, labels_train, labels_test

