# ML-AI-python
The Repository will act as a Pathway that i will be following to get myself introduced and comfortable with ML and AI concepts.

# Basics Notes
Some steps usefull -> load data. check columns, info, draw snspairplots for smaller data, check distributions ,find correlation and heat map and look for correlation with the variable we want to predict or is dependent.

## Bias Varience and Regularization:
Bias-Varience tells us if or a model is underfit or overfit. **Bias** refers to the error introduced by approximating a real-world problem, which may be extremely complicated, by a simplified model. A high-bias model is one that oversimplifies the problem and misses the underlying patterns in the data. This can lead to systematic errors consistently deviating predictions from the true values.
**Variance** refers to the model's sensitivity to small fluctuations in the training data.
A high-variance model captures random noise in the training data and does not generalize well to new, unseen data.
High-variance models tend to overfit the training data, meaning they fit the noise in the data instead of the underlying patterns.

**Regularization** is a technique used in machine learning to prevent overfitting by adding a penalty term to the model's objective function. It helps to control the complexity of a model and prevents it from fitting the noise in the training data. Regularization is particularly useful when dealing with high-dimensional data or when there are many features in the model. 3 types L1, L2 and Elastic(L1+L2).

**Cross-Validation** ??

## Feature Scaling:
- **Standardization-** scaling the data so that it has a mean of 0 and SD of 1.
- **Normalization-** Rescale values between 0 and 1.
**Feature scaling is done only to training data and not the complete data set!**

# Learning process for a ML Model - 

**get data -> clean data -> divide data from training, validation and testing -> modle training and building-> model testing -> model deployment.**

* **Training Data** - data used by model to fit the parameters accordingly.
* **Validation Data** - (A part of Test Data) check the performance of the Traning data and accordinglt adjust the parameters.
* **Testing Data** - Evaluate true/final preformance of the model (Real world performance).

# Evaluating Performance - Classification

In classification the model can be either correct or incorrect

## Confusion Matrix -> organizing predicted values to real values.
* **True Positive(TP)** -> actual and predicted True
* **False Negative(FN, Type2 error)** -> actual True predicted False
* **Flase Positive(FP, Type1 error)** -> actual False predicted True
* **True Negative(TN)** -> actual and predicted False

## Different Evalution Parameters:
* **Accuracy -** number of correct prediction / number of total prediction. USeful when the target classess are well balanced(50-50 in case of binary classification).
### -> For unbalanced data set we go for recall and precision.
* **Recall -** number of true positives/number of true positives + false negatives. ability of a model to find ALL RELEVENT cases within a dataset.  
* **Precision -** number of true positives/number of true positives + false positives. ability of a model to find ONLY THE RELEVENT data points. ** Expresses the poropotion of the data points our model says was relevent actually were relevent.
* **F1 Score -** combination of both precision and recall. Harmonic mean?? of precision and recall: 2* (precision*recall)/(precision+recall)

# Evaluating Performance - Regression

In regression we have numerical value predicition. Happens on validation and testing data.

## Different Evalution Parameters:
* **Mean(Avg) Absolute(Mod) Error** - avg of |true values - pridicted| <- donsn't work good for data with large errors 
* **Mean Square Error** - avg of square of (true values - pridicted) <-changes unit of the error adds square to it 
* **Root Mean Square Error** - root of (avg of square of (true values - pridicted)) <- Most popular

# Types of Learning :
## Supervised learning - 
Algo trained using labeled examples. Inputs where the desired results are known. Algorithm is given set of input and the correct output, algo learns by comparing its actual output with the correct to find error and learn from them and modify itself.

**METHODS like** classification, regssion, prediction and gradient boosting -> SL uses patterns to predict label on additional unlabeled data. SL -> used where historical data predicts likely future events. 

## Unsupervised Learning - 
used against data that has no historical labels. System not tols the right answers, goal is to explor the data and find some structure within. 

**METHODS like** self-organizing maps, neareast-neighbou mapping, k-means clustering and singular value decompositon.

# Models and Projects
## 1. Linear Regression
### Tips
Load data, check head, check info() and describe().
Draw plots, scatter, jointplot(seaborn), pairplot(between all features), distplay, heatmaps(IMP).
After pairplot we can draw some conclusion on plots and feature with max correlation and we can plot linear plot to find out.

Now, split the data fram 1. drop all non numerical column 2. Y= dependent feature 3. X= all other independent feature.
After split, make test train split and then import the model.
Once the model is fit we can check the coefficient to see if out plot conclusion is correct.

After fitting the data, check intercepts and coeficient. To analyse we can make a dataframe for each column and its coefficient .from which we can remove some relationship. We can now predict values to see the output data and then we can compare it with the Y_test or actual predicted value by plotting a histogram of the resuidals. The histogram should show a normal distribution if the model choice is right. Another better method to see if LR is the correct model is to plot residual VS actual y values. <- the last one is usefull in multiple independent variable dataset i.e. multiple x values/ features.

## 2. Polynomial Regression 
### Tips
We can the Polynomial regression model and fit and transform the data. By transform we make combination and see polynomial relationship of independent varibales with self and each other by specifying the degree.
One way to find out what degree perform well is that we train the model for different degree and then we plot the RMSE for train and test y values against the degree. 
