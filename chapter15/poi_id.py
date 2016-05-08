#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Just take as many features as possible for now
# Omitting email_address as it should be irrelevant to POI status 
features_list = ['poi','salary','deferral_payments', 'total_payments',
        'loan_advances', 'bonus', 'restricted_stock_deferred',
        'deferred_income', 'total_stock_value', 'expenses',
        'exercised_stock_options', 'other', 'long_term_incentive',
        'restricted_stock', 'director_fees', 'to_messages', 
        'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
        'shared_receipt_with_poi'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Determine total POIs vs. total records to get class skew
pois = [data_dict[k] for k in data_dict if data_dict[k]['poi'] == True]
print "Total number of POIs:", len(pois)
print "Total records: ", len(data_dict)


### Task 2: Remove outliers

# We don't actually remove anything for now because POIs are relatively rare and
# may take extreme values for certain features, e.g. salary

# Remove names as these aren't helpful for feature engineering
records = [data_dict[k] for k in data_dict]

# Get dict of raw values for each input feature
featureSets = { f: [records[k][f] for k in range(0,len(records))] for f in features_list }
featureSetsNoNaN = { f:[w for w in v if w != 'NaN'] for f,v in featureSets.items() }

# Collect some stats on our features: min, max, mean, median 
stats = { f: (round(min(featureSetsNoNaN[f])), 
              round(max(featureSetsNoNaN[f])), 
              round(np.mean(featureSetsNoNaN[f])),
              round(np.median(featureSetsNoNaN[f]))) for f in features_list }
print "Feature stats: ", stats

# Determine %NaN per feature to see if imputed values would be representative/useful
percentNaNs = { f: 100*(float(len(featureSets[f]))-len(featureSetsNoNaN[f]))/len(featureSets[f]) 
                for f in featureSets }
print "Percent NaNs: ", percentNaNs 

# Lots of NaNs and median/mean don't match because we have skewed data. Probably
# a better idea to map our features into [0,1] and set all NaNs to 0.5 which is
# the mean for the uniform prior in the new space


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

minPerFeature = { f: stats[f][0] for f in features_list }
print "Min per feature: ", minPerFeature

maxPerFeature = { f: stats[f][1] for f in features_list }
print "Max per feature: ", maxPerFeature

# MinMax Feature Scaling
def newFeature(feature, value):
    if feature == 'poi':
        return value
    if value == 'NaN':
        return 0.5
    else:
        return float((value-minPerFeature[feature])) / (maxPerFeature[feature]-minPerFeature[feature])

# Helper function for feature renaming
def newName(feature):
    if feature == 'poi':
        return feature
    else:
        return feature+'_new'

# Rescaled features dict and list
newFeatures = {k: {newName(f): newFeature(f,data_dict[k][f]) for f in features_list } 
                for k in data_dict}
newFeatures_list = ['poi'] + [f for f in newFeatures[newFeatures.keys()[0]] if f != 'poi']

# Set my_dataset so the stuff downstream works
my_dataset = newFeatures 
features_list = newFeatures_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
