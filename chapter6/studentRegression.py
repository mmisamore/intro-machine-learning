def studentReg(ages_train, net_worths_train):
    
    from sklearn.linear_model import LinearRegression
    
    reg = LinearRegression()
    reg.fit(ages_train, net_worths_train)
    
    return reg

