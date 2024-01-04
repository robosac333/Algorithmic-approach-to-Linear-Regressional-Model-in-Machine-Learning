#!usr/env/bin python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
# from tkinter import Tk, filedialog


# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


# # Open a file dialog for the user to select a file
# root = Tk()
# root.withdraw()  # Hide the main window
# file_path = filedialog.askopenfilename(title="ex1data1", filetypes=[("Txt Files", "*.txt")])
file_path = "/home/sj/Downloads/ex1data1.txt"
# root.destroy()

# Check if a file was selected
if file_path:
    # Read the CSV file
    df = pd.read_csv(file_path, names=['population', 'profit'])
    print(df.head())
    # Continue with your analysis or visualization code here
else:
    print("No file selected.")


fig, ax_lst = plt.subplots(1, 1)  # a figure with a 1x1 grid of Axes
fig.suptitle('profit vs population plot')  # Add a title so we know which it is
ax_lst.xaxis.set_label_text('Population in 10,000s')
ax_lst.yaxis.set_label_text('Profit in $10,000s')

# Complete following scatter plot code
ax_lst.scatter(df.population.values,df.profit.values,marker='.',c='r',label='Training Data')
ax_lst.legend()
# plt.show()

p=df.population.values
m=len(p)
X=np.array([np.ones(m),p]) #add first column to ones to X
X=np.transpose(X)
y=df.profit.values
theta=np.zeros(2)

def computeCost(X, y, theta):
    # COMPUTECOST Compute cost for linear regression
    #   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    #  parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = len(y) # number of training examples

    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    J = 1/(2*m) * sum(np.square(np.dot(X, theta) - y))


    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    #GRADIENTDESCENT Performs gradient descent to learn theta
    #   GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha
    #   Returns updated values of theta after num_iters iterations

    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros(num_iters)

    for iter in range(num_iters):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector theta.

        hypo = np.dot(X, theta)
        X0 = X[:, 0]
        X1 = X[:, 1]



        t1 = theta[0] - alpha * (1 / m) * sum(X0*(np.dot(X, theta) - y));
        t2 = theta[1] - alpha * (1 / m) * sum(X1*(np.dot(X, theta) - y));

        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        theta[0] = t1
        theta[1] = t2



        # ============================================================
        # Save the cost J in every iteration
        J_history[iter]=computeCost(X, y, theta)

        print('#',iter,'theta',theta,'J',J_history[iter])

    return theta

iterations = 5000
alpha = 0.01
theta = gradientDescent(X, y, theta, alpha, iterations)

print('With theta = [0,0], cost computed J = %10.2f'%computeCost(X,y,[0,0]), '(Expected cost = 32.07 approx.)')
print('With theta = [-1,2], cost computed J = %10.2f'%computeCost(X,y,[-1,2]), '(Expected cost = 54.24 approx.)')
print('Theta found by gradient descent= %10.4f , %10.4f'%(theta[0],theta[1]),'(Expected values: -3.8953, 1.1930)')

y_predicted=np.dot(X,np.transpose(theta))
# complete following code
ax_lst.plot(X[:,1],y_predicted,c='b',label='Linear Regression')
ax_lst.legend()
plt.show()