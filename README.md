Implemented Regressional Analysis on a dataset of 97 values and adjusted the values of slope and intercept through 5000 iterations to obtain a regression line using the standard Regression algorithms.
Implementation of Regressional Analysis without using any Machine Learning python Libraries such as Scikit, TensorFlow or Pytorch.

**Step 1 : Linear Regression with one variable**
Used this data to help select which city to expand to next. The file
ex1data1.txt contains the dataset for our linear regression problem. The first column is the
population of a city and the second column is the profit of a food truck in that city. A negative
value for profit indicatesa loss.

**Step 2 : Overview of Gradient Descent**
With the available data points and assuming the initial values of learning rate, slope and intercept, we device a cost function

![image](https://github.com/robosac333/Algorithmic-approach-to-Linear-Regressional-Model-in-Machine-Learning/assets/143353582/0163deda-49df-4e7e-b992-23d7994a258a)

where the hypothesis hθ(xi) is given by the linear model.

Note: We store each example as a row in the the X array.Totake into
account the intercept term (θ0), we add an additional first column to X and set it to all
ones. This allows us to treat θ0 as simply another ‘feature’.

![image](https://github.com/robosac333/Algorithmic-approach-to-Linear-Regressional-Model-in-Machine-Learning/assets/143353582/5aa7db64-d2ba-4e93-b14f-c1e4b1a258c4)


**Step 3 : Update the values of parameters in large number of iteration at a specific learning rate**
Witheachstepofgradientdescent, yourparametersθj come closer tothe optimal
values that will achieve the lowest cost J(θ).

![image](https://github.com/robosac333/Algorithmic-approach-to-Linear-Regressional-Model-in-Machine-Learning/assets/143353582/41459221-7c67-46fc-bc4f-c9d205d12af7)

**Step 4 : Obtain the Linear Regression line by plotting the data points and their predicted output y values**
Used Matplotlib to plot the values

![image](https://github.com/robosac333/Algorithmic-approach-to-Linear-Regressional-Model-in-Machine-Learning/assets/143353582/2c41f200-4b98-4e75-ba09-a83925601c2b)


