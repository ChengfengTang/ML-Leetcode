# Kmeans algorithm for network traffic analysis
"""
Input1:
3                 #N = 3
50 25 30          #Initial Centroids
60 15 60
25 75 90
3                 #Iterations = 3
9                 #N = 9
50 25 30
30 50 30
60 15 60
25 75 90
10 05 60
26 15 30
32 67.5 90
80 7.5 60
20 100 90

Output:
35.33 30.00 30.00
50.00 9.17 60.00
25.67 80.83 90.00

Input 2:
3
50 20 30
60 10 60
180 180 180
3
8
50 20 30
30 50 30
60 10 60
25 75 90
100 5 60
30 60 90
80 10 60
180 180 180

Output:
40.00 35.00 30.00
59.00 32.00 72.00
180.00 180.00 180.00
"""
import math
input1 = "3\n50 25 30\n60 15 60\n25 75 90\n3\n9\n50 25 30\n30 50 30\n60 15 60\n25 75 90\n10 05 60\n26 15 30\n32 67.5 90\n80 7.5 60\n20 100 90"

def dataset(raw):
    lines = raw.split("\n")
    centroidsN = int(lines[0])
    centroids = []
    iterator = 1
    while iterator < centroidsN + 1:
        centroids.append([float(lines[iterator].split(" ")[0]), float(lines[iterator].split(" ")[1]), float(lines[iterator].split(" ")[2])])
        iterator += 1
    iterations = int(lines[iterator])
    iterator += 2
    data = []
    while iterator < len(lines):
        data.append([float(lines[iterator].split(" ")[0]), float(lines[iterator].split(" ")[1]), float(lines[iterator].split(" ")[2])])
        iterator += 1
    return centroidsN, iterations, data, centroids


def kmeansraw(data, centroids, iterations):
    for i in range(iterations):
        newCentroids = [[] for _ in range(len(centroids))] # list of lists, each list is a cluster for a centroid
        for point in data: # for each data point, find the closest centroid
            minDist = float("inf")
            minCentroid = 0
            for i in range(len(centroids)):
                dist = math.dist(point, centroids[i])
                if dist < minDist:
                    minDist = dist
                    minCentroid = i
            newCentroids[minCentroid].append(point) # add the data point to the closest centroid's cluster
        #print(newCentroids)
        for i in range(len(newCentroids)): # for each centroid, calculate the new centroid
            if len(newCentroids[i]) > 0:
                temp = [0.0,0.0,0.0] # temporary list to store the new centroid
                for x in range(len(newCentroids[i])):
                    temp[0] += newCentroids[i][x][0]
                    temp[1] += newCentroids[i][x][1]
                    temp[2] += newCentroids[i][x][2]
                temp[0] /= len(newCentroids[i])
                temp[1] /= len(newCentroids[i])
                temp[2] /= len(newCentroids[i])
                newCentroids[i] = temp
        centroids = newCentroids # update the centroids with the new centroids
        #print(centroids)
    return centroids

input1 = "3\n50 25 30\n60 15 60\n25 75 90\n3\n9\n50 25 30\n30 50 30\n60 15 60\n25 75 90\n10 05 60\n26 15 30\n32 67.5 90\n80 7.5 60\n20 100 90"
N, iterations, data, centroids = dataset(input1)
#print(iterations)
#print(data)
#print(centroids)
print(kmeansraw(data, centroids, iterations))

input2 = "3\n50 20 30\n60 10 60\n180 180 180\n3\n8\n50 20 30\n30 50 30\n60 10 60\n25 75 90\n100 5 60\n30 60 90\n80 10 60\n180 180 180"
N, iterations, data, centroids = dataset(input2)
#print(iterations)
#print(data)
#print(centroids)
print(kmeansraw(data, centroids, iterations))