import csv

"""
split csv file into inputs_list and labels_list
"""

# open file and read in rows
file = open('jsonoutput.csv', encoding='utf-16')
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

print(inputs_list[0])
print(labels_list[0])