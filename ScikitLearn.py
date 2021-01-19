# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:16:45 2020

@author: 014174

MODEL SELECTION - predicting compliance

"""
# =============================================================================
# Package Import
# =============================================================================
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, ComplementNB,MultinomialNB, CategoricalNB
from sklearn.metrics import confusion_matrix
from sklearn import svm

import pickle
import csv

# =============================================================================
# Data import
# =============================================================================
os.chdir("")
df = pd.read_csv("TFIDF.csv") #TFIDF
df2 = pd.read_csv("full_final_merged_firms.csv") #RAW


# =============================================================================
# Data Cleaning
# =============================================================================
df = df.drop(columns = ["Unnamed: 0", "adj_name" ])
col_names = df.columns
df_np = df.to_numpy()

df2 = df2.drop(columns = ["Unnamed: 0", "adj_name" ])
col_names2 = df2.columns
df2_np = df2.to_numpy()

# =============================================================================
# Train-Test Split
# =============================================================================
y = df_np[:,0] # same y for both datasets
X1 = df_np[:,1:]
X2 = df2_np[:,1:]

X1_train, X1_test, y_train, y_test = train_test_split(X1,y, test_size = 0.33,
                                                    random_state = 42)

X2_train, X2_test, y_train, y_test = train_test_split(X2,y, test_size = 0.33,
                                                    random_state = 42)
# =============================================================================
# Naive Bays - multinomial classifier
#
#Most likely classifier to be succesful. Doing this to set a baseline before 
#moving on to NN. Will use non-TFIDF data with this not to throw off probabilities.
# =============================================================================

NB_model = MultinomialNB()
NB_model.fit(X2_train, y_train)
NB_model.predict(X2_test)
nb_prediction = NB_model.predict(X2_test)
print("Naive Bays results:")
print(confusion_matrix(y_test,nb_prediction))
print("\n")
#accuracy is 78% on its first attempt 


model = GaussianNB()
model.fit(X2_train, y_train)
guassian_predictions = model.predict(X2_test)
print("Guassian results:")
print(confusion_matrix(y_test, guassian_predictions))
print("\n")
#Performs much worse than Multinomial NB - leave this here.

model = MultinomialNB()
model.fit(X1_train, y_train)
model.predict(X1_test)
nb_prediction = model.predict(X1_test)
print("Naive Bays TFIDF results:")
print(confusion_matrix(y_test,nb_prediction))
print("\n")
#Using TFIDF reduces accuracy - as expected. Continue with normal data

model = ComplementNB()
model.fit(X2_train, y_train)
model.predict(X2_test)
nb_prediction = model.predict(X2_test)
print("Complement NB results:")
print(confusion_matrix(y_test,nb_prediction))
print("\n")
#Complement NB works as well as multinomial and distribution seems more equal
#This makes sense as its design is for imbalanced datasets

model = svm.SVC(decision_function_shape='ovo')
model.fit(X2_train, y_train)
model.predict(X2_test)
nb_prediction = model.predict(X2_test)
print("SVM results:")
print(confusion_matrix(y_test,nb_prediction))
print("\n")

# =============================================================================
# The best model therefore seems to be the complment NB or the simple MNNB.
# They perform to an identical accuracy of 78% but with different predictions
# at this very early stage the complementNB seems the most appropriate to use 
# because of the equality of its choices - but this is slightly pedantic.
# =============================================================================

# =============================================================================
# 
# # save the model to disk
# filename = 'finalized_NB_model.sav'
# pickle.dump(NB_model, open(filename, 'wb'))
# col_names = list(df.columns)[1:]
# 
# with open('col_names.csv','w', encoding='utf-8') as result_file:
#     wr = csv.writer(result_file, dialect='excel')
#     wr.writerows(col_names)
#     
# =============================================================================
