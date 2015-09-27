# Machine Learning - Course Project
Michael Michonski  
September 26, 2015  

##Executive Summary  
The following analysis contains the process of building a model to predict correct and incorrect movements in weightlifting using Groupware's Human Activity Recognition dataset.

  
  
  
##Data Cleaning  


To load the data:


```r
setwd("/Users/rossc/Downloads")
weight.training <- read.csv(file="pml-training.csv", na.strings = c("NA", "N/A", "#DIV/0!"))
```

To explore features of the data:


```r
str(weight.training)
summary(weight.training)
```

There are many N/A's in the dataset. Thus we look more closely to see how many variables contian a significant number of N/A's missing with:


```r
colSums(!is.na(weight.training))
```

Upon closer examination it appears these variables are not essential to the raw data but rather averages, minimums, and maximums of the raw data. They also contain a large number of N/A's. Therefore we can get rid of them with:


```r
wt.compact <- weight.training[,colSums(is.na(weight.training)) < 0.5*nrow(weight.training)]
```

Furthermore, we can see that there are a few columns at the beginning related to timestamping the datapoints, which cannot possibly have any help in predicting what type of movement the wearer is performing. Thus we get rid of them with:


```r
wt.compact <- wt.compact[,8:60]
```

Now we can begin looking at which variables might have a strong correlation with each other in order to reduce the dimensions of our dataset a bit more using:


```r
set.seed(5)
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Warning: package 'ggplot2' was built under R version 3.1.3
```

```r
wt.cor <- cor(wt.compact[,1:52])
wt.high.cor <- findCorrelation(wt.cor, cutoff = 0.75)
sort.hc <- sort(wt.high.cor)
wt.reduced <- wt.compact[,-c(sort.hc)]
```


##Model Building  

Now we partition the data:


```r
inTrain <- createDataPartition(wt.reduced$classe, p = 0.75, list = FALSE)
training <- wt.reduced[inTrain, ]
testing <- wt.reduced[-inTrain, ]
```

Then we set a model to fit the data. Here we will use random forets due to large and most likely non-linear model needed to fit the data. Furthermore, random forests have cross-validation built into them so that no k-fold cross validation is needed.


```r
library(randomForest)
modelFit <- randomForest(classe~., data = training)
modelFit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 5
## 
##         OOB estimate of  error rate: 0.6%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4183    1    1    0    0 0.0004778973
## B    9 2834    3    0    2 0.0049157303
## C    0   20 2529   18    0 0.0148032723
## D    0    0   25 2385    2 0.0111940299
## E    0    0    3    4 2699 0.0025868441
```

Finally we test it on the testing set we partitioned earlier and look at a plot of the error vs number of trees:


```r
confusionMatrix(testing$classe, predict(modelFit, newdata = testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393    1    0    0    1
##          B    5  942    2    0    0
##          C    0    7  842    6    0
##          D    0    0    6  798    0
##          E    0    0    0    2  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9939          
##                  95% CI : (0.9913, 0.9959)
##     No Information Rate : 0.2851          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9923          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9916   0.9906   0.9901   0.9989
## Specificity            0.9994   0.9982   0.9968   0.9985   0.9995
## Pos Pred Value         0.9986   0.9926   0.9848   0.9925   0.9978
## Neg Pred Value         0.9986   0.9980   0.9980   0.9980   0.9998
## Prevalence             0.2851   0.1937   0.1733   0.1644   0.1835
## Detection Rate         0.2841   0.1921   0.1717   0.1627   0.1833
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9979   0.9949   0.9937   0.9943   0.9992
```

```r
plot(modelFit, main = "Visualizing Model Construction")
```

![](Machine_Learning_-_Course_Project_-_Weight_Lifting_Dataset_Analysis_files/figure-html/unnamed-chunk-9-1.png) 

It appears we could have used a much smaller number of trees, however with a resubstitution error of 0.6% and an out of sample error of 0.61%, it bodes well that the out of sample error for the final testing set will be around 0.61% if not a little greator.



