# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 20:53:28 2022

@author: hp
"""
#importing the libraries
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#opening file
all_data = open('E:\Sem 7\IML\webKb data\combined.txt', 'r')
#bag of words
from collections import Counter
def word_count(fname):
        with open(fname) as all_data:
                return Counter(all_data.read().split())
print("Number of words in the file :",word_count("E:\Sem 7\IML\webKb data\combined.txt"))
#no need to print the bag of words as it is not required in finding accuracy.
labels = { "student":3,"faculty":2,"project":1,"course":0}
# making empty lists and assigning variables
output_data = list() #list containing all the lines with their respective labels
data_under_classification = list()  #list containing data to be trained and tested
accuracy_naive_bayes = list()
accuracy_random_forest = list()
tfidf = TfidfVectorizer()
naive_bayes = GaussianNB()
random_forest = RandomForestClassifier()
training_count = 3
for var in range(training_count):
    all_lines = all_data.readlines()
    random.shuffle(all_lines) #shuffling all lines on purpose
    #filling out the two lists
    for every_line in all_lines: 
        output_data += [labels[every_line.split("\t")[0]]]
        data_under_classification += [every_line.split("\t")[1:][-1]]
    tfidf_train_vector = tfidf.fit_transform(data_under_classification)
    #printing tf-idf
    #print("For training example",var+1, "tidf is \n",tfidf_train_vector)
    x_trained, x_testing , y_trained, y_testing = train_test_split(data_under_classification,output_data, train_size=0.7,random_state=0)
    training_vectors = tfidf.transform(x_trained)
    testing_vectors = tfidf.transform(x_testing)
    #naive_bayes
    naive_bayes.fit(training_vectors.toarray(),y_trained)    
    predicted_y = naive_bayes.predict(testing_vectors.toarray())
    accuracy_naive_bayes += [accuracy_score(y_testing,predicted_y)]
    #print("Accuracy of Naive Bayes Classification for training no.", var+1,"is",accuracy_naive_bayes[var])
    #random_forest
    random_forest.fit(training_vectors,y_trained)    
    predicted_y = random_forest.predict(testing_vectors)
    accuracy_random_forest += [accuracy_score(y_testing,predicted_y)]
    #print("Accuracy of Random Forest Classification for training no.", var+1,"is",accuracy_random_forest[var])
    
    