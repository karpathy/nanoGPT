#https://www.kaggle.com/datasets/sulphatet/twitter-financial-news?resource=download

import csv

with open('train_data.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    i = 0
    txt_file = open('input.txt', 'w')
    for row in reader:
        string = row[0].split('http')[0]
        # skip text and label 
        if i > 0: 
            txt_file.write(string+'\n')
        i+=1

    print("Added %d string of text", i-1)
