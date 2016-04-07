#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    # Extract into a set of tuples and compute error for each
    zipped = zip(ages[:,0].tolist(), predictions[:,0].tolist(), net_worths[:,0].tolist()) 
    errors = map(lambda (a,p,nw): (a,nw,abs(p-nw)), zipped) 

    # Determine how many elements to take for 90 percent
    ninetyPercent = len(errors) - len(errors)/10

    # Sort by increasing errors and return 90 percent
    sortedErrors = sorted(errors, key=lambda (a,nw,e): e)
    cleaned_data = sortedErrors[0:ninetyPercent]

    return cleaned_data

