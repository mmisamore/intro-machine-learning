#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

# Remove the 'TOTAL' entry, which is an outlier
data_dict.pop('TOTAL',0)
data = featureFormat(data_dict, features)

# Find a few more outliers
outliers = { k:v for k,v in data_dict.iteritems() 
            if v['salary'] != 'NaN' 
            and v['salary'] > 10**6 
            and v['bonus'] > 5*10**6 }.keys()
print outliers

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

