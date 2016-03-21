#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# Cut down training data to speed things up
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

# Instantiate SVM model and fit
clf = SVC(kernel="rbf", C=10000.0)
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

# Use the model to make some predictions
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

# Determine and print the accuracy
print accuracy_score(labels_test,pred)

# Some sample predictions
print "Samples for pred[10], pred[26], pred[50]",
print pred[10], pred[26], pred[50]

# Number of Chris class
print "Number of Chris class",
print len(pred[pred == 1])

