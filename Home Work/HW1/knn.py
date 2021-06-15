import math
import numpy as np  
from download_mnist import load
import operator  
import time
# classify using kNN  
#x_train = np.load('../x_train.npy')
#y_train = np.load('../y_train.npy')
#x_test = np.load('../x_test.npy')
#y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)
def kNNClassify(newInput, dataSet, labels, k): 
    
    result=[]
    
    # compute L2 distance for all test and train samples
    distances = np.zeros((len(newInput), len(dataSet)))
    for i in range(len(newInput)):
    	for j in range(len(dataSet)):
    		distance = np.sqrt(np.sum((newInput[i]-dataSet[j])**2))
    		distances[i, j] = distance
    # decide which class the test samples belong in
    
    for i in range(len(newInput)):
    	label_value = np.zeros(10)
    	knn_indices = np.argsort(distances[i])[:k]
    	for j in range(len(knn_indices)):
    		label = labels[knn_indices[j]]
    		label_value[label]+=1
    	result.append(np.argmax(label_value))

    return result

start_time = time.time()
outputlabels=kNNClassify(x_test[0:20],x_train,y_train,10)
result = y_test[0:20] - outputlabels
result = (1 - np.count_nonzero(result)/len(outputlabels))
print ("---classification accuracy for knn on mnist: %s ---" %result)
print ("---execution time: %s seconds ---" % (time.time() - start_time))