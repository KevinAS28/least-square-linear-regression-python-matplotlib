import matplotlib.pyplot as plt 
import numpy as np
import random
from threading import Thread
"""
REGRESSION LINEAR: LEAST SQUARE METHOD

https://medium.com/deep-math-machine-learning-ai/chapter-1-complete-linear-regression-with-math-25b2639dde23

my skeleton model

lets say i have these equations
a = 1
b = 2
c = 3
d = 4
e = 5
a + b + c = 6
a + b + d = 7
a + b + e = 8

"""

def simple(x_input):
	plt.figure()
	plt.title("Simple Least Square Regression Linear")
	x = np.array([i + (5 if (i%2==0) else 10) for i in range(100)])
	y = np.array([i + (5 if (i%1==0) else 10) for i in range(100)])

	x_mean = sum(x)/len(x)
	y_mean = sum(y)/len(y)
	m = sum((x[i]-x_mean)*(y[i]-y_mean) for i in range(len(x)))/sum((x[i]-x_mean)**2 for i in range(len(x)))
	b = y_mean - m*x_mean

	plt.scatter(x, y, color = "m", marker = "o", s = 30)


	y_new = m*x_input+b
	plt.scatter([x_input], [y_new], color = "g", marker = "o", s = 30)

	x_show = [min(x), x_input]
	y_show = [-b, y_new]

	# new_x = np.array(list(m*i+b for i in x))
	plt.plot(x_show, y_show,color = "r", linewidth=3)
	

def multi_part(x_input, z_input):
	plt.figure()
	plt.title("Positive Least Square Regression Linear")
	# x = np.array([3, 9, 5])
	# z = np.array([2, 8, 4])
	# y = np.array([5, 17, 9])
	
	x = np.array([i + (5 if (i%2==0) else 10) for i in range(100)])
	z = np.array([i + (6.25 if (i%1==0) else 8.75) for i in range(100)])	
	y = np.array([i + (5 if (i%1==0) else 10) for i in range(100)]) 

	# x = np.array([random.randrange(0, 50) for i in range(0, 50)])
	# z = np.array([random.randrange(0, 50) for i in range(0, 50)])
	# y = np.array([ (x[i]+z[i]) for i in range(len(x))])

	val = [[x, "g"], [z, "b"]]
	x_mean = sum(x)/len(x)
	z_mean = sum(z)/len(z)
	y_mean = sum(y)/len(y)
	m0 = sum((x[i]-x_mean)*(y[i]-y_mean) for i in range(len(x)))/sum((x[i]-x_mean)**2 for i in range(len(x)))
	m1 = sum((z[i]-z_mean)*(y[i]-y_mean) for i in range(len(z)))/sum((z[i]-z_mean)**2 for i in range(len(z)))
	#b0 = y_mean - m0*x_mean
	#b1 = y_mean - m1*z_mean
	m = (m0+m1)/2
	else_mean = (z_mean+x_mean)/2
	b = y_mean - m*else_mean #((y_mean - m0*x_mean) + (y_mean - m1*z_mean))/2
	
	y_new = b + (m0*x_input + m1*z_input)/2 #((b0+m0*x_input)+(b1+m1*z_input))/2
	
	for i in range(len(x)):
		plt.scatter(x[i]+z[i], y[i], color = "g" if (i%2==0) else "b", marker = "o", s = 30)

	print("input: ", x_input+z_input, " b: ", b, " y: ", y_new)
	x_show = [min(x[i]+z[i] for i in range(len(x))), x_input+z_input]

	#for -b, its not original, but i found its good enough for start of the line
	y_show = [-b, y_new]
	plt.plot(x_show, y_show,color = "r", linewidth=3)
	plt.scatter(x_input+z_input, y_new, color = "r" , marker = "o", s = 50)
	
def neg_multi_part(x_input, z_input):
	plt.figure()
	plt.title ("Negative Least Square Regression Linear")
	# x = np.array([3, 9, 5])
	# z = np.array([2, 8, 4])
	# y = np.array([5, 17, 9])
	
	x = np.array([-i + (-5 if (i%2==0) else -10) for i in range(100)])
	z = np.array([-i + (-6.25 if (i%1==0) else -8.75) for i in range(100)])	
	y = np.array([-i + (-5 if (i%1==0) else -10) for i in range(100)]) 

	# x = np.array([random.randrange(0, 50) for i in range(0, 50)])
	# z = np.array([random.randrange(0, 50) for i in range(0, 50)])
	# y = np.array([ (x[i]+z[i]) for i in range(len(x))])

	val = [[x, "g"], [z, "b"]]
	x_mean = sum(x)/len(x)
	z_mean = sum(z)/len(z)
	y_mean = sum(y)/len(y)
	m0 = sum((x[i]-x_mean)*(y[i]-y_mean) for i in range(len(x)))/sum((x[i]-x_mean)**2 for i in range(len(x)))
	m1 = sum((z[i]-z_mean)*(y[i]-y_mean) for i in range(len(z)))/sum((z[i]-z_mean)**2 for i in range(len(z)))
	#b0 = y_mean - m0*x_mean
	#b1 = y_mean - m1*z_mean
	m = (m0+m1)/2
	else_mean = (z_mean+x_mean)/2
	b = -(y_mean - m*else_mean) #((y_mean - m0*x_mean) + (y_mean - m1*z_mean))/2
	
	#B is negative, to fit the data. change it if necessary
	y_new = (-b) + (m0*x_input + m1*z_input)/2 #((b0+m0*x_input)+(b1+m1*z_input))/2
	

	for i in range(len(x)):
		plt.scatter(x[i]+z[i], y[i], color = "g" if (i%2==0) else "b", marker = "o", s = 30)

	print("input: ", x_input+z_input, " b: ", b, " y: ", y_new)
	x_show = [max(x[i]+z[i] for i in range(len(x))), x_input+z_input]
	y_show = [b, y_new]
	plt.plot(x_show, y_show,color = "r", linewidth=3)
	plt.scatter(x_input+z_input, y_new, color = "r" , marker = "o", s = 50)
	

def main():
	multi_part(50, 60)
	neg_multi_part(-75, -90)
	simple(85)
	plt.show()



if __name__=="__main__":
	main()
