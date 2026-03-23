# 3 dimensional data classification using KNN
"""
Input 
10 3           #K = 3, N = 10
0.5 0.3 0.4 0  #D=3, C
0.6 0.2 0.5 0  
0.4 0.3 0.3 0  
0.7 0.4 0.6 0  
2.1 2.3 2.2 1  
2.3 2.2 2.4 1  
2.2 2.4 2.3 1  
4.5 4.3 4.4 2  
4.4 4.5 4.6 2  
4.6 4.4 4.5 2  
2.2 2.1 2.3    #new X
Expected Out: 
1

Input 
12 4  
1 0.9 1 1  
0.8 0.9 0.7 1  
1.3 1 1.2 1  
1.2 0.9 1 1  
2 2.2 2.1 2  
2.3 2.2 2 2 2  
2 2.2 1.9 2  
1.9 2.2 2.1 2  
3.1 3.1 3 3  
2.8 2.9 3.1 3  
2.9 3 3.2 3  
3.1 3 3.1 3  
2.2 1.2 1.9
Expected Out: 
2
"""
from collections import defaultdict
from math import dist
from re import X


input1 = "10 3\n0.5 0.3 0.4 0\n0.6 0.2 0.5 0\n0.4 0.3 0.3 0\n0.7 0.4 0.6 0\n2.1 2.3 2.2 1\n2.3 2.2 2.4 1\n2.2 2.4 2.3 1\n4.5 4.3 4.4 2\n4.4 4.5 4.6 2\n4.6 4.4 4.5 2\n2.2 2.1 2.3"
input2 = "12 4\n1 0.9 1 1\n0.8 0.9 0.7 1\n1.3 1 1.2 1\n1.2 0.9 1 1\n2 2.2 2.1 2\n2.3 2.2 2 2\n2 2.2 1.9 2\n1.9 2.2 2.1 2\n3.1 3.1 3 3\n2.8 2.9 3.1 3\n2.9 3 3.2 3\n3.1 3 3.1 3\n2.2 1.2 1.9"

def dataset(raw):
    lines = raw.split("\n")
    data = []
    first = lines[0].split(" ")
    N = int(first[0])
    K = int(first[1])
    for i in range(1, len(lines) - 1):
        data.append([float(x) for x in lines[i].split(" ")])
    newX = [float(x) for x in lines[-1].split(" ")]
    return N, K, data, newX

def knn(data, newX, D, K):
    nearestX = [] # list of tuples, each tuple is a data point and its distance to the new data point
    for x in data:
        nearestX.append((x,dist(x[0:D], newX))) # add the data point and its distance to the new data point to the list
    nearestX.sort(key=lambda x: x[1]) # sort the list by the distance
    res = [x[0][D] for x in nearestX[0:K]] # get the labels of the nearest K data points
    return max(set(res), key=res.count) # return the class with the most frequent label

N, K, data, newX = dataset(input1) # N: number of data points, K: number of classes, data: list of data points, newX: new data point
D = len(data[0]) - 1  # features; last column is label
#print(data)
print(knn(data, newX, D, K))
N, K, data, newX = dataset(input2) # N: number of data points, K: number of classes, data: list of data points, newX: new data point
D = len(data[0]) - 1  # features; last column is label
#print(data)
print(knn(data, newX, D, K))