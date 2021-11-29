import csv
import sklearn
import sklearn.model_selection
import math

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
        inputs_list.append(inputs)
        labels_list.append(labels)

    labels_list = list(map(round,labels_list))
    # print(inputs_list[:10])
    print(labels_list[:10])

    # Use sklearn to do train and test split 0f 0.25 | 0.75
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(inputs_list, labels_list, test_size=0.25, random_state=0)


    print('Train size: ', len(X_train))
    print('Test size: ', len(X_test))
    print('Total:', len(X_train) + len(X_test))

    return X_train, X_test, y_train, y_test

