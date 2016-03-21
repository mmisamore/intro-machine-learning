def classify(features_train, labels_train):   
    # import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    # create and fit the classifier
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    
    # return the fit classifier
    return clf


