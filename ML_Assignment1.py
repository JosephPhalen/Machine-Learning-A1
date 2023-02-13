import numpy as np
import csv
x = []
y = []
#Cleaning Data
with open('iris.csv',newline='\n') as data:
    for line in data:
        temp = line.split(',')
        if 'Iris-setosa' in temp[4]:
            y.append(1)
        else:
            y.append(-1)
        temp.pop(4)
        
        for i in range(len(temp)):
            temp[i] = temp[i].replace('\"','')
            temp[i] = float(temp[i])
        temp.insert(0,1)
        x.append(temp) 

#Modify Weights      
def adjust(weights,data,y):
    x = np.dot(y,data)
    new = np.add(weights,x)
    return new
#Classify
def classify(weights,x):
    h = []
    weights = np.transpose(weights)
    #print(weights)
    for i in range(len(x)):
        
        num = np.matmul(weights,x[i])
        if num < 0:
            num = -1
        else:
            num = 1
        h.append(num)
    return h

weights = [1.0,1.0,1.0,1.0,1.0]
h = []
finished = False
h = classify(weights,x)

#Learning Algorithm
while(finished != True):
    changed = False
    for i in range(120):
        if h[i] != y[i]:
            weights = adjust(weights,x[i],y[i])
            changed = True
            break
    h = classify(weights,x)
    #print(weights)
    if changed == False:
        finished = True
print('Finished')
print('Learned Hypothesis:')
print(weights)

print('Test:\nh(t) y(t)')

#Testing data
h2 = classify(weights,x)
sum = 0 
for i in range(150):
    if h2[i] == y[i]:
        print(h2[i], '  ', y[i])
        sum +=1 
        
print(sum, ' out of 150 correct.')
