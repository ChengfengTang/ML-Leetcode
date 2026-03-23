# Linear regression to predict phone price
"""
Input 
10                  #N = 10
86 99 20 3595       #D=3, Column 4 is label
175 171 90 6596 
194 42 47 4691 
192 172 26 5927 
44 20 168 4169 
61 138 64 4348 
161 42 85 4791 
197 181 99 7126 
170 55 95 5208 
26 158 142 5231
2                   #new X
159 135 173 
120 144 59

Input 2:
4
30 23 24 1999 
55 53 46 2999 
68 85 78 3999 
113 90 103 4999
1
126 114 143

"""
from sklearn.linear_model import LinearRegression
import numpy as np

input1 = "10\n86 99 20 3595\n175 171 90 6596\n194 42 47 4691\n192 172 26 5927\n44 20 168 4169\n61 138 64 4348\n161 42 85 4791\n197 181 99 7126\n170 55 95 5208\n26 158 142 5231\n2\n159 135 173\n120 144 59"
input2 = "4\n30 23 24 1999\n55 53 46 2999\n68 85 78 3999\n113 90 103 4999\n1\n126 114 143"

def dataset(raw):
    lines = raw.split("\n")
    data = []
    labels = []
    newX = []
    N = int(lines[0])
    for i in range(1, N + 1):
        temp = lines[i].split(" ")
        data.append([float(temp[0]), float(temp[1]), float(temp[2])])
        labels.append(float(temp[3]))
    for i in range(N + 2, len(lines)):
        newX.append([float(lines[i].split(" ")[0]), float(lines[i].split(" ")[1]), float(lines[i].split(" ")[2])])
    return N, data, labels, newX

def LR(data, labels, newX):
    model = LinearRegression()
    model.fit(data, labels)
    return model.predict(newX)

def LRraw(data, labels, newX):
    # Closed-form linear regression (normal equation) with bias term. We model: y = Xw + b -> y = [1, X] * [b, w]^T
    x = np.asarray(data, dtype=float)  # (N, D)
    y = np.asarray(labels, dtype=float)  # (N,)
    x_new = np.asarray(newX, dtype=float)  # (M, D)
    ones = np.ones((x.shape[0], 1), dtype=float) # (N, 1)
    X = np.concatenate([ones, x], axis=1)  # (N, D+1) -> augmented X matrix
    ones_new = np.ones((x_new.shape[0], 1), dtype=float) # (M, 1)
    X_new = np.concatenate([ones_new, x_new], axis=1)  # (M, D+1) -> augmented X_new matrix

    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return X_new @ theta  # (M,) 

N, data, labels, newX = dataset(input1)
print(LR(data, labels, newX))
N, data, labels, newX = dataset(input1)
print(LRraw(data, labels, newX))
N, data, labels, newX = dataset(input2)
print(LR(data, labels, newX))
N, data, labels, newX = dataset(input2)
print(LRraw(data, labels, newX))