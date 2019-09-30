import numpy as np
from succinctly.datasets import get_dataset, linearly_separable as ls

'''

I followed the code from the eBook 'Support Vector Machines Succinctly'
When it came to updating the weights, the code here is different from Luis Serrano did in 
his video on this topic. The code here used the addition and subtraction of vectors to seemingly rotate the vector while
Luis Serrano rotated the line wrt the data points ( both wrongly and correctly classified ) by adjusting the weights
proportionate to the coordinates of those points.

This suggests to me perhaps different variations of the PLA. More to be discovered. To be continued...


'''


def perceptron_learning_algorithm(X, y):
    w = np.random.rand(3) # Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1)
    misclassified_examples: object = predict(hypothesis, X, y, w) # returns an array of misclassified training data

    # as long as there's misclassified examples
    # we will keep updating our weights
    while misclassified_examples.any():
        x , true_y = pick_one_from(misclassified_examples, X, y)
        w = w + x * true_y # update the weights
        misclassified_examples = predict(hypothesis, X, y, w)

    return w

def hypothesis(x, w):
    return np.sign(np.dot(w,x))

    # Make predictions on all data points
    # and return the ones that are mis-classified.

def predict(hypothesis_function, X, y, w):
    predictions = np.apply_along_axis(hypothesis_function, 1, X, w)
    # apply hypothesis function to every row ( axis = 1) of X i.e every data point
    # w is additional argument of the input function
    misclassified = X[y != predictions]
    # here y != predictions returns the indices of which the elements in y array are not equal to
    # the corresponding element in the predictions array, hence giving the elements (our data points)
    # in X which gives an incorrect prediciton
    return misclassified




def pick_one_from(misclassified_examples, X, y):
    # Pick one misclassified example randomly
    np.random.shuffle(misclassified_examples)
    x = misclassified_examples[0]
    # and return it with its true label
    index = np.where(np.all(X == x, axis=1))
    return x, y[index]

# algorithmic code above
# test run on toy data below

np.random.seed(88) #initialize the pseudo-random number generator
X, y =get_dataset(ls.get_training_examples)

# transform x into an array of augmented vectors
# from w'x = 0 to w'x + b = 0

X_augmented = np.c_[np.ones(X.shape[0]), X]

w = perceptron_learning_algorithm(X_augmented, y)

print(w)