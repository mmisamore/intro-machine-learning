import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf1 = DecisionTreeClassifier(min_samples_split=2)
clf1.fit(features_train, labels_train)
pred1 = clf1.predict(features_test)
acc_min_samples_split_2 = accuracy_score(labels_test, pred1)

clf2 = DecisionTreeClassifier(min_samples_split=50)
clf2.fit(features_train, labels_train)
pred2 = clf2.predict(features_test)
acc_min_samples_split_50 = accuracy_score(labels_test, pred2)

def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}
