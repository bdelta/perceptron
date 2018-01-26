from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
from fractions import Fraction
import random as rd

#This is a neural network with 2 layers
#2 input nodes in the first layer
#1 output node in the second layer
#input data
data = [[3,  1.5, 1],
		[2,  1,   0],
		[4,  1.5, 1],
		[3,  1,   0],
		[3.5, .5, 1],
		[2,  .5,  0],
		[5.5, 1,  1],
		[1,   1,  0]]

#Change the value of mystery flower to test data points
mystery_flower = [rd.random()*6, rd.random()*2]

def pred(m1, m2, w1, w2, b):
	return m1*w1 + m2*w2 + b

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_p(x):
	return sigmoid(x)*(1-sigmoid(x))

def cost(pred, target):
	return np.square(sigmoid(pred) - target)

def cost_p(pred, target):
	return 2*(sigmoid(pred) - target)


#Scatter plot of the points
plt.figure(1)
plt.grid()
plt.axis([0, 6, 0, 4])
for i in range(len(data)):
	point = data[i]
	color = 'r'
	if point[2] == 0:
		color = 'b'
	plt.scatter(point[0],point[1],c=color)


learning_rate = 0.2


#Initilize values
w1 = 0
w2 = 0
b = 0

for i in range(1000):
	cost_sum_w1 = 0
	cost_sum_w2 = 0
	cost_sum_b = 0
	#Compute the cost function and its derivative
	for i in range(len(data)):
		point = data[i]

		z = pred(point[0], point[1], w1, w2, b)

		cost_sum_w1 += cost_p(sigmoid(z),point[2])*sigmoid_p(z)*point[0]
		cost_sum_w2 += cost_p(sigmoid(z),point[2])*sigmoid_p(z)*point[1]
		cost_sum_b += cost_p(sigmoid(z),point[2])*sigmoid_p(z)

	w1 -= learning_rate*Fraction(1,len(data))*cost_sum_w1
	w2 -= learning_rate*Fraction(1,len(data))*cost_sum_w2
	b -= learning_rate*Fraction(1,len(data))*cost_sum_b

	#print(w1)
	#print(w2)
	#print(b)
	#This should be decreasing
	#print(cost(z,point[2]))

x = sigmoid(pred(mystery_flower[0],mystery_flower[1],w1,w2,b))
color = 'r'
result = 'red'
if x < 0.5:
	color = 'b'
	result = 'blue'

#Triangle represents the mystery marker
plt.scatter(mystery_flower[0],mystery_flower[1], c = color, marker='v', s=50)
print("The result is " + repr(x) + ' which is ' + result)

plt.show()