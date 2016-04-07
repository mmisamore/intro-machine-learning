import numpy
import matplotlib.pyplot as plt

from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(ages_train, net_worths_train)

# Predict net worth
km_net_worth = reg.predict([27])[0][0] ### fill in the line of code to get the right value

# Get slope for model
slope = reg.coef_[0][0]

# Get intercept for model
intercept = reg.intercept_[0]

# get the score on test data
test_score = reg.score(ages_train,net_worths_train)

# get the score on the training data
training_score = reg.score(ages_test,net_worths_test)

def submitFit():
    # all of the values in the returned dictionary are expected to be
    # numbers for the purpose of the grader.
    return {"networth":km_net_worth,
            "slope":slope,
            "intercept":intercept,
            "stats on test":test_score,
            "stats on training": training_score}

