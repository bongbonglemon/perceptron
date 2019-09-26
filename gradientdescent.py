from typing import List

import numpy as np

'''
Comments:

I can improve on my variable naming. I feel not being familiar with the relevant math
terms was one reason for my incompetency. I could also work towards better presentation. Siraj's
code actually shows the gradient descent working to make the slight slope adjustments with every
iteration.

'''

# find line of best fit (y=mx + b) using gradient descent

# plan
# make a first guess
# find sum of squared differences
# differentiate with respect to b and m
# then use step size = slope x learning rate
# then new b = old b - step size
# repeat

learning_rate = 0.0001
num_of_iterations = 1000
points = np.genfromtxt("data.csv", delimiter=",")

def run():
    b = 0  # initial b guess
    m = 0  # initial m guess
    for i in range(num_of_iterations):
        step_size_b, step_size_m = step_size(b, m)
        b -= step_size_b
        m -= step_size_m
    return b, m


def step_size(b, m):
    slope_b, slope_m = find_slope(b, m)
    step_size_b = slope_b * learning_rate
    step_size_m = slope_m * learning_rate
    return step_size_b, step_size_m


def find_slope(b, m):
    slope_b = 0
    slope_m = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i,0] # first column
        y = points[i,1] # second column

        slope_b += -2*(y-(m*x+b))
        slope_m += -2*x*(y-(m*x+b))

        '''
        slope_b += -(2/N) * (y - (m * x + b))
        slope_m += -(2/N) * x * (y - (m * x + b))
        
        KIV: This gives the same answer as Siraj's but I don't know why he divides
        every term by the number of points.
        '''

    return slope_b, slope_m


b,m = run()
print(b,m)

