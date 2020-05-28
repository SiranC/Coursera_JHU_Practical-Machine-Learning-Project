---
title: "Practical Machine Learning Project"
author: "SiranC"
output: 
        html_document:
                keep_md: true
---



## Overview

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, you will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [here](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har). (see the section on the Weight Lifting Exercise Dataset).

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

#### Data 

The training data for this project are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv). The test data are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv). The data for this project come from this [source](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har).

## Data Process

First, download the datasets and read them in the memory.


```r
library(data.table)
# download the data
fileUrl_train = 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
fileUrl_test = 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
if (!file.exists('./pml-training.csv')){
        download.file(fileUrl_train,destfile = './pml-training.csv')
}
if (!file.exists('./pml-testing.csv')){
        download.file(fileUrl_test,destfile = './pml-testing.csv')
}
# read the data
training <- read.csv('pml-training.csv', header=TRUE)
testing <- read.csv('pml-testing.csv', header=TRUE)
```

The training data had 19622 observations and 160 variables. The distribution of five types of execution of the Unilateral Dumbbell Biceps Curl exercise, representing by A, B, C, D, and E, showed below:


```r
dim(training)
```

```
## [1] 19622   160
```

```r
summary(training$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

Next, some columns have a lot of missing values, some are unrelated to predicting, and some are nearly zero variance, we will clean the datasets by removing those columns.


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
#remove first 7 unrelated columns
training <- training[,8:length(training)]
#remove the columns with NAs and missing value
training <- training[,colSums(is.na(training))==0]
#remove the columns near zero variance
nzvCol <- nearZeroVar(training)
training <- training[,-nzvCol]
```

## Model Develop

Now we can select the model and train it.

### Cross-validation

We first split the training dataset into new train and validation sets.


```r
set.seed(123)
inTrain <- createDataPartition(y = training$classe,
                               p = 0.70, list = FALSE)
train <- training[inTrain,]
validation <- training[-inTrain,]
```

### Model Train

We will use the random forest algorithm to train the data since it is one of the most effective and accurate machine learning models. This ensemble technique uses bootstrap aggregation to sample the data with the replacement multiple times, then the decision trees will be generated in parallel from each subsample. The final predictions is based on combining decisions from submodels. Therefore, the random forest can reduce the bias since it not heavilly rely on specific features and it can prevent overfitting. 


```r
set.seed(123)
modFit <- train(classe~., data = train, method = 'rf')
```

### Model Validation

Let's use the validation dataset to test how our model performs by calculating the cross-validation accuracy.


```r
predVali <- predict(modFit, validation)
cfmVali <- confusionMatrix(predVali, validation$classe)
cfmVali
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    6    0    0    0
##          B    1 1125    5    0    0
##          C    0    8 1017   10    4
##          D    0    0    4  954    5
##          E    0    0    0    0 1073
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9927          
##                  95% CI : (0.9902, 0.9947)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9908          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9877   0.9912   0.9896   0.9917
## Specificity            0.9986   0.9987   0.9955   0.9982   1.0000
## Pos Pred Value         0.9964   0.9947   0.9788   0.9907   1.0000
## Neg Pred Value         0.9998   0.9971   0.9981   0.9980   0.9981
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1912   0.1728   0.1621   0.1823
## Detection Prevalence   0.2853   0.1922   0.1766   0.1636   0.1823
## Balanced Accuracy      0.9990   0.9932   0.9934   0.9939   0.9958
```

We can see that the accuracy is 0.9927 which is pretty good. Out-of-sample(or out-of-bag OOB) error rate can be seen in model **modFit** output **finalModel**. The OOB is 0.0066.


```r
modFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.66%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3898    6    1    0    1 0.002048131
## B   16 2636    6    0    0 0.008276900
## C    0   12 2373   11    0 0.009599332
## D    0    0   26 2224    2 0.012433393
## E    0    1    4    5 2515 0.003960396
```

## Cases Prediction

Now we use out model to predict 20 test cases.


```r
predTest <- predict(modFit, testing[,8:length(testing)])
predTest
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
