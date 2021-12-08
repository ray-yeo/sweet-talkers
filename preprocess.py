import csv
import sklearn
import sklearn.model_selection
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
STOPWORDS = set(stopwords.words('english'))

vocab_size = 3000
embedding_dim = 64
max_length = 437
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
testing_portion = .1

"""
split csv file into inputs_list and labels_list
"""
def roundToNearestInt(x):
    return round(float(x))

def twoLabel(x):
    if (int(float(x)) >= 4):
        return 1
    else:
        return 0

def get_data(filename = 'jsonoutput.csv', two_label = True):

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

    if (two_label): 
        for row in rows:
            inputs = row[5]
            labels = row[4]
            inputs_list.append(inputs)
            labels_list.append(labels)

        labels_list = list(map(twoLabel,labels_list))
        # test labels list correctly work with 1s and 0s
        print(labels_list[:10])

        # Use sklearn to do train and test split 0f 0.25 | 0.75
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(inputs_list, labels_list, test_size=0.25, random_state=0)


        print('Train size: ', len(X_train))
        print('Test size: ', len(X_test))
        print('Total:', len(X_train) + len(X_test))

        return X_train, X_test, y_train, y_test
    else: 
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
        train_inputs, test_inputs, train_labels, test_labels = sklearn.model_selection.train_test_split(inputs_list, labels_list, test_size=testing_portion, random_state=0)

        tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
        
        tokenizer.fit_on_texts(train_inputs)
        word_index = tokenizer.word_index

        # see top 20 most frequent words
        words_dict = dict(list(word_index.items())[0:20])
        print(words_dict)

        train_sequences = tokenizer.texts_to_sequences(train_inputs)

        train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        
        test_sequences = tokenizer.texts_to_sequences(test_inputs)
        test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


        # print('Train size: ', len(train_inputs))
        # print('Test size: ', len(test_inputs))
        # print('Total:', len(train_inputs) + len(test_inputs))

        return train_padded, test_padded, train_labels, test_labels

