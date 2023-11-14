 # ML-AI-python
The Repository will act as a Pathway that i will be following to get myself introduced and comfortable with ML and AI concepts.

# Basics Notes
Some steps usefull -> load data. check columns, info, draw snspairplots for smaller data, check distributions ,find correlation and heat map and look for correlation with the variable we want to predict or is dependent.

## Bias Varience and Regularization:
**Bias-Varience** tells us if or a model is underfit or overfit.   
**Bias** refers to the error introduced by approximating a real-world problem, which may be extremely complicated, by a simplified model. **A high-bias model is one that oversimplifies/underfits** the problem and misses the underlying patterns in the data. This can lead to systematic errors consistently deviating predictions from the true values.  
**Variance** refers to the model's sensitivity to small fluctuations in the training data. A high-variance model captures random noise in the training data and does not generalize well to new, unseen data. **High-variance models tend to overfit the training data**, meaning they fit the noise in the data instead of the underlying patterns.  

**Regularization** is a technique used in machine learning to prevent overfitting by adding a penalty term to the model's objective function. It helps to control the complexity of a model and prevents it from fitting the noise in the training data. Regularization is particularly useful when dealing with high-dimensional data or when there are many features in the model. 3 types L1, L2 and Elastic(L1+L2). For regularization we first fit_transform the data if necessary and split the data into test and training split. we scale the training data only!(sklearn-StandardScalar)

### Conaidering line equation as : $` \hat{y} = \hat{\beta}_{0} + \sum \limits _{j=1} ^{p} x_{j}\hat{\beta}_{j} `$
### RMS = $` 1/n \sqrt{ \sum \limits _{j=1} ^{n} (y_{j} - \hat{y}_{j})^2} `$
## RSS(Sum of square Resuidal) = $` \sum \limits _{j=1} ^{n} (y_{j} - \hat{y}_{j})^2 `$

- ### L2 / Ridge Regression
Goal is to reduce over fitting of the model by adding an extra panelty term in the error or the **RSS** of the model. Given by : **RSS** + $` \lambda \sum \limits _{j=1} ^{p} \beta_{j}^2  `$;  
where $`\beta`$ is the cofficient of the independent variables and $`\lambda`$ tells us the severity of the penalty and can be from 0 to $`\infty`$  
L2 is like adding some bias so that to reduce the varience and that in turns reduce the over fitting.

- ### L1 / Lasso Regression
Mostly used of rfeature selection  
Goal is to reduce over fitting of the model by adding an extra panelty term which is equal to the absolute value of the slope in the error or the **RSS** of the model. Given by : **RSS** + $` \lambda \sum \limits _{j=1} ^{p} |\beta_{j}|  `$;  
Lasso might actually make some of the coefficient zero for a large enough $`\alpha`${according to skilearn}/ $`\lambda`$ value. This makes the model more simpler. Also its not necessary it will make the model better then L2 always!
  - **IMPORTANT IN L1** - While training, take care of eps, n_alpha, cv, max_inter parameters

**Cross-Validation** : is usually done K-folds and is a cycle of train and validate unitl the model is trained on the entire training data set. check GridSearch() and normal methods with CV.

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
* **Recall -** (Sensitiivity) number of true positives/number of true positives + false negatives. ability of a model to find ALL RELEVENT cases within a dataset.  
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
## Regression
### 1.1. Linear Regression
#### Tips
Load data, check head, check info() and describe().
Draw plots, scatter, jointplot(seaborn), pairplot(between all features), distplay, heatmaps(IMP).
After pairplot we can draw some conclusion on plots and feature with max correlation and we can plot linear plot to find out.

Now, split the data fram 1. drop all non numerical column 2. Y= dependent feature 3. X= all other independent feature.
After split, make test train split and then import the model.
Once the model is fit we can check the coefficient to see if out plot conclusion is correct.

After fitting the data, check intercepts and coeficient. To analyse we can make a dataframe for each column and its coefficient .from which we can remove some relationship. We can now predict values to see the output data and then we can compare it with the Y_test or actual predicted value by plotting a histogram of the resuidals. The histogram should show a normal distribution if the model choice is right. Another better method to see if LR is the correct model is to plot residual VS actual y values. <- the last one is usefull in multiple independent variable dataset i.e. multiple x values/ features.

### 1.2. Polynomial Regression 
#### Tips
We can the Polynomial regression model and fit and transform the data. Only the independent variables (X) are fit_transfrorm. By transform we make combination and see polynomial relationship of independent varibales with self and each other by specifying the degree.
One way to find out what degree perform well is that we train the model for different degree and then we plot the RMSE for train and test y values against the degree.  
## Logistic Regression  
Even though the name says regression it is a classification. Linear regressions are converted to Logistic by converting continous targets into catagorial through **Discretization**  
Classification algorithms - along with classification provides probabilistic data eg. probability od data being in a specific catagory.  
Error metrics of linear regression wont be useful here

- **Logistic Function** - A function we can say that maps all the values of x (+ || -) between 0 and 1.  
In linear regression the fitting of curve was done using square of resuidal sum but in logistic similar cannot be done, so , we use a concept named maximum likelihood.
It is binary classification. At the end we can find the probablity or likelyhood of a data being in a said class or not. Accoeding to my understanding why it is called regression is because 1. its being supervised learning, 2. converting continous independent variable into a binary probablity.

## KNN (K-Neareast Neighbour)  
Again mostly used for classification, performs poorly in regression models. Intitution is it selects the class to which majority of the points it is near to. In case of a tie in majority point there are various ways to break the tie. Sckit learn slects the first class near to it in case of tie.  
Scaling the data is neceassary and a goood idea when having multiple feature since distace can be more in one feature and less in another may create a bias.

## K-Means Clustering  
Basically we initialize a "K" i.e. number of clusters we want. Then we randomly select "k" cluster point in data set and then assign points to these cluster point on the basis of nearest distance to them. Then we take avg of each cluster and set the new cluster points to these averages and we repeat the above process again so then we get the proper cluster formation.  
One way to find the vule for K is using elbow method(From what i understood is visually plotting the graph for the error metrics and choosing a point where it flats out "creating an elbow") and the error metrics here would be Sum of Squared Error (SSE) which is sum of squared distances of each point in the cluster to its centroid.  

## Hierarichal Clustering  
We dont have to specify anynumber of clustering in the beginning. It visualizes on own which will then help us to decide number of cluster. 2 approches -> 
- i - **Agglomerative Approch:** each point begain as own cluster and then joined together  
In the above approch the main question is how do we measure proximity/distance/similarity between two clusters? there are 4 possible ways
  1. MIN(Single Linkage) : Distance b/w closest point for two cluster
  2. MAX(Complete Linkage) : Distance b/w farthest point for two cluster
  3. Group Average : Avg distance of all the points in both clusters
  4. Distance Between centroids : Distance b/w centroids of two cluster  
- ii - **Divisive Approch:** everything as one cluster and then divided into individual clusters

## SVM - Support Vector Mechine  
Hyperplane(it is a flat plane that is a (N-1) dimension in a N dimentional space) to create a seperation between classess.  
A **Maximal Margin Classifier** is a seperator eg a hyperplane that maximizes a margin between different classess. 
A **Support Vector Classifier/Soft Margins** allowing miss classification within the margins to get an over all good classification model **<- Concept still not clear?**
There is stilla cases where our above two models will fail, what if there exist no hyper plane which cannot seperate our classess ? there here we sill use SVM and how that is achieved is using kernels.  
**Kernals** are used to project our points to a higher dimantion where it can be then seperated by a hyperplane. To avoid this expensive computation we use something knwon as Kernel Trick.  
In kernel trick we use Dot Product **(similarity)** a . b = |a||b|cos(@)  
Can use gridSearchCV for various values of C <- Soft Boundry and Gamma <- for kernal trick  
**SVM most fo the time do NEEED grid search to find the best parameter!!**

## Naive Bayes and NLP
It follows the Bayes expresssion in probablity where A and B are events : $` P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} `$  
​NB is mostly used in NLP. In NLP there is somthing known as count vectorization i.e. counting the occurance of each word in a given class.
### Feature Extraction from Text  
2 methods :  
- **Count Vectorization** : Vocubulary of all the words used in all documents and then create a Document Term Matrix **(DTM)** ~similar to; just for example! how we create hot encoded values columns. Treats every word as a feature with their frequency count as a strength of the feature/words. This leads to ab problem where frequebcy of any word is zero and it leads to the probablity of a new the new word to be classified to be wrongly assigned to another class. One solution to solve this would be adding one in all the count vectors so tha we have no 0 left and another method is mentioned below.  
- **TF-IDF** (Term Frequency-Incerse Document Frequency) : TF-IDF is used to solve the problem that can be casued by the above method. IDf includes the occurance of term in each document too. tf is simple it is the count of the term is is multipled by idf which is calculated by total no of documents divded by the number of documents that countains a word and calculating the logarithm.

## Trees
The splitting cretria in a tree is also known as information gain.entropy and information gain are the mathematical methods of choosing the best split.
### 1. Decision Tree
Pruninig : a method to shorten the leaf to avoid over fitting.  
**Gini impurity**- Mathematical measurement of how "pure" the information in a data set is. By pure it means the data containing more data for a specific class. eg red and blue data the less red and more blue means the data is more pure wrt to blue data. We use gini impurity in decision tree to minimize gini impurity / maximize gini purity at the leaf node. <- **IMPORTANT!!**  
We dont scale the data before spliting in case of decision tree classification.  
Prone to over fitting.  

### 2. Random Forest
Working with multiple decision tree. from a feature set N, every time m samples (m <= N) are chosen at random and a split is made from those m samples. 

## Boosting Methods
Boosting is also known as a meta learning algorithm since it can be applied to any learnign model. But commonly applied to decision trees. Its a method and not an model on its own. Aggregative a bunch of weak model into an ensemble model. BOosting is also known as an optimizing algorithm on top of a cost function.
### 1. AdaBoosting (Adaptive)  
Creating ensemble of multiple weak model. In case of a decision tree creating multiple stump (i.e. a node with two leafs). Trees are built in series. Why Series? basically its how the alpha or weights are propogated. 
### 2. Gradient Boosting  
Similar to adaboost but ise resuidal error for learning. Difference from Adaboost - Larger trees are allowed, learning coeffiecent same for all weak learners, training on resuidal of the previosu model. for classification we use **logit?** function to get the error.
