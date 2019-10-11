'''
Perceptron network for classification of wine based on quality
High Quality:1
Low Quality:-1
'''
import numpy as np
import pandas as pd
from perceptron import Perceptron

csv_path = 'wine_quality.csv'
df = pd.read_csv(csv_path)
print("dataframe created")
x=np.array(df.iloc[:,0:11])         #to convert dtype from float64
y=df.iloc[:,11]
y=np.where(y > 5,1,-1)

epoch=input("Enter the no. of epochs:")
l_r=float(input("Enter the desired learning rate:"))
p = Perceptron(x.shape[1],epoch,l_r)
print "Training the Perceptron..."
p.train(x, y)
print("___Final weights___")
print(p.weights)
count=0
for i in range(x.shape[0]):
    z=p.predict(x[i])
    if z==y[i]:
        count+=1
per=float(count)/x.shape[0]*100
print "Accuracy rate:",per

print "Testing the Perceptron..."
test=np.array(df.iloc[1000:1010,0:11])
result=df.iloc[1000:1010,11]
result=np.where(result > 5,1,-1)
print "___Desired Output___"
print result
print "___Actual Output___"
for i in range(10):
    print(p.predict(test[i]))
