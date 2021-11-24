"""
Adapted from https://www.geeksforgeeks.org/convert-json-to-csv-in-python/
"""


import json
import csv
 
with open('data_request_fall_2021_raymond_geireann.json') as json_file:
    jsondata = json.load(json_file)

print("json file read successfully")
 
data_file = open('jsonoutput.csv', 'w', newline='')
csv_writer = csv.writer(data_file)
 
count = 0
for data in jsondata:
    if count == 0:
        header = data.keys()
        csv_writer.writerow(header)
        count += 1
    csv_writer.writerow(data.values())
 
data_file.close()

print("done")