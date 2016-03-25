#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
# print sorted(enron_data.keys())

# Persons of interest
pois = dict(enumerate([enron_data[k] for k in enron_data if enron_data[k]['poi'] == True]))

# print enron_data['PRENTICE JAMES']
# print enron_data['COLWELL WESLEY']

# print enron_data['LAY KENNETH L']
# print enron_data['SKILLING JEFFREY K']
# print enron_data['FASTOW ANDREW S']

# print sorted(enron_data.keys())

# Quantified salary
knownSalaries = [enron_data[k] for k in enron_data if enron_data[k]['salary'] != 'NaN']
knownEmails   = [enron_data[k] for k in enron_data if enron_data[k]['email_address'] != 'NaN']
# print len(knownSalaries), len(knownEmails)

# print len([k for k in enron_data if enron_data[k]['total_payments'] == 'NaN'])/float(len(enron_data))
# print [pois[k]['total_payments'] for k in pois]
# print len([k for k in pois if pois[k]['total_payments'] == 'NaN'])/float(len(pois))

# Adding 10 new POIs with NaN for total_payments
# print len(enron_data)+10
# print len([k for k in enron_data if enron_data[k]['total_payments'] == 'NaN'])+10

print len(pois)+10
print len([k for k in pois if pois[k]['total_payments'] == 'NaN'])+10

