""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    minArr = float(min(arr))
    maxArr = float(max(arr))
    try:
        newArr = [(x-minArr)/(maxArr-minArr) for x in arr]
    except:
        newArr = arr
    return newArr

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)


