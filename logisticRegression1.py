# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 12:18:54 2018

@author: Ammar
"""

import csv

import matplotlib.pyplot
import pylab
import numpy as np
import math
import copy

#reading the data from the file
with open('ex2data1.csv', 'r') as f:
  reader = csv.reader(f)
  data = list(reader)

Features=[]

n=len(data[0])-1

#as many features as n-1
for i in range(0,n):
    Features.append([])


#populate the list using data
for i in range(0,len(data)):
    for j in range (0,n):
        Features[j].append(float(data[i][j]))
        
        
      
        
#Making and initialzing Theetas
Theetas=[]
for i in range (0,n+1):
    Theetas.append(float(0))
    
#separating the 0's and 1's
admittedBoolean=[]
for i in range(0,len(data)):
  admittedBoolean.append(float(data[i][n]))
  

for i in range(0,len(data)):
    if admittedBoolean[i]==0:
        matplotlib.pyplot.scatter(Features[0][i],Features[1][i],c='red',marker='o')
    elif admittedBoolean[i]==1:
        matplotlib.pyplot.scatter(Features[0][i],Features[1][i],c='green',marker='x')

matplotlib.pyplot.xlabel("Exam 1 Score")
matplotlib.pyplot.ylabel("Exam 2 Score")


#normalizing the features

means=[]
stds=[]

for i in range(0,n):
    mean=np.mean(Features[i])
    std=np.std(Features[i])
    means.append(float(mean))
    stds.append(float(std))
    for j in range(len(Features[i])):
        Features[i][j]=(Features[i][j]-mean)/std  
      

#function to find GofX
def findGOfX(TheetaTX):
  
    denom=1+math.exp(-(TheetaTX))
    
    return (1/denom)
    
#function to find hOfX for prediction purposes    
def find_h_of_x_for_Evaluation(testTheetas=[],testFeatures=[]):
    
    matTheetas=np.asmatrix(testTheetas)
    matFeatures=np.asmatrix(testFeatures)
    hofx=np.matmul(matTheetas,matFeatures.transpose())
    
   
    return hofx

#function to find hOfX for gradient descent purpose   
def find_h_of_x_at_Index(index,Theetas=[],Features=[]):
   
    x=[]

    x.append(float(1))

    for i in range(0,len(Features)):
        x.append(Features[i][index])
        
    matTheetas=np.asmatrix(Theetas)
    matFeatures=np.asmatrix(x)
    hofx=np.matmul(matTheetas,matFeatures.transpose())
   
    return hofx
    
    
   
#cost function
def costFunction(m,Theetas=[], Features=[],admittedBoolean=[]):
    cost=0
    for i in range(0,m):
        firstTerm=(-1*admittedBoolean[i])*(math.log(findGOfX(find_h_of_x_at_Index(i,Theetas,Features)) ))
        secondTerm=(1-admittedBoolean[i])*(math.log(1-findGOfX(find_h_of_x_at_Index(i,Theetas,Features)) ))
        finalTerm=firstTerm-secondTerm
        cost=cost+finalTerm
    
    cost=(cost/m)
    
    return cost



#function to update theetas
def updateTheeta(k,m,Theetas=[],Features=[],admittedBoolean=[]):
   sum=0
   alpha=0.001
   for i in range(0,m):
       firstTerm=findGOfX(find_h_of_x_at_Index(i,Theetas,Features))
       secondTerm=admittedBoolean[i]
       
       x=[]

       x.append(float(1))

       for l in range(0,len(Features)):
           x.append(Features[l][i])
       
       
       finalTerm=(firstTerm-secondTerm)*x[k]
       sum=sum+finalTerm
       
   return (alpha*sum)


tempTheetas=copy.copy(Theetas)

#applying gradient descent and updating the theetas
for i in range(0,4000):    
    for j in range(0,len(tempTheetas)):
        tempTheetas[j]=tempTheetas[j]-updateTheeta(j,len(data),Theetas,Features,admittedBoolean)
        
    Theetas=copy.copy(tempTheetas)
    tempTheetas=copy.copy(Theetas)
   
 
print("Cost after optimizing theetas:")    
print(costFunction(len(data),Theetas,Features,admittedBoolean))
print("Optimized theetas:")   
print(Theetas)   



#Plotting the boundary line

x = np.linspace(30, 100, 50)

y = -(Theetas[0] + Theetas[1]*x)/Theetas[2]+130
matplotlib.pyplot.plot(x, y)

    
    

#Evaluating the test example given in the assignment

Exam1Score=45
Exam2Score=85

normalizedExam1Score=(Exam1Score-means[0])/stds[0]
normalizedExam2Score=(Exam2Score-means[1])/stds[1]
testFeatures=[]
testFeatures.append(float(1))
testFeatures.append(float(normalizedExam1Score))
testFeatures.append(float(normalizedExam2Score))

print("Probability of test example:")
print(findGOfX(find_h_of_x_for_Evaluation(Theetas,testFeatures)))

       
        
    
    


    

    
    
    
