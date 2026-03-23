#
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
"""

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

iterations, data, centroids = dataset(input1)
print(iterations)
print(data)
print(centroids)

def kmeansraw(data, centroids, iterations):
    pass