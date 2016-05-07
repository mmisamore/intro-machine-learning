#!/usr/bin/python

"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

# Get features and labels to feed the classifier
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# Split for cross-validation
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    features, labels, test_size=0.3, random_state=42
)

# Fit based on training data and check accuracy with test data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf2 = DecisionTreeClassifier()
clf2.fit(features_train,labels_train)
pred = clf2.predict(features_test)

# print sum(pred)
# 4 POIs predicted

# print len(pred)
# 29 total people in test set

# print sum(labels_test)
# Number of POIs actually in test set: 4.0

# Accuracy if we predict no POIs at all (skewed class)
# print (29-4.0)/29
# 0.862

# Compute true positives
true_positives = sum([1 for i in range(0,len(pred)) if pred[i] == 1 and labels_test[i] == 1])
print "True positives: ", true_positives

# Compute false positives
false_positives = sum([1 for i in range(0,len(pred)) if pred[i] == 1 and labels_test[i] == 0]) 
print "False positives: ", false_positives

# Compute false negatives
false_negatives = sum([1 for i in range(0,len(pred)) if pred[i] == 0 and labels_test[i] == 1]) 
print "False negatives: ", false_negatives

from sklearn.metrics import precision_score, recall_score

print "Sklearn precision: ", precision_score(labels_test,pred)
print "Sklearn recall: ", recall_score(labels_test,pred)
print "Sklearn accuracy: ", accuracy_score(labels_test, pred)

# For the quiz:
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

true_pos = sum([1 for i in range(0,len(predictions)) if predictions[i] == 1 and true_labels[i] == 1])
true_neg = sum([1 for i in range(0,len(predictions)) if predictions[i] == 0 and true_labels[i] == 0])
false_pos = sum([1 for i in range(0,len(predictions)) if predictions[i] == 1 and true_labels[i] == 0])
false_neg = sum([1 for i in range(0,len(predictions)) if predictions[i] == 0 and true_labels[i] == 1])

print "true_pos, true_neg, false_pos, false_neg: ", true_pos, true_neg, false_pos, false_neg
print "Precision: ", float(true_pos)/(true_pos + false_pos)
print "Recall: ", float(true_pos)/(true_pos + false_neg)

