
#########################################################################################################################################
## Objective: Machine learning of Titanic data's survival classification with ensemble methods (boosted decision tree)                  #
## Data source: titanic data set split into train and test                                                                              #
## Please install "bst" package: install.packages("bst") for Gradient Boosting                                                          #
#########################################################################################################################################

##load the library
library(bst)

##DATA EXPLORATION AND CLEANING
##Load the Titanic data in R
titanic.data <- read.csv("C:/Users/muckam/Desktop/DataScienceBootcamp/Datasets/titanic.csv", header=TRUE)
## explore the data set
dim(titanic.data)
str(titanic.data)
summary(titanic.data)
##Ignore the PassengerID, Name, Ticket, and Cabin
titanic.data <- titanic.data[, -c(1, 4, 9, 11)]
##The bst default settings require that binary target classes have the values {-1, 1}
##map 0 -> -1 in Survived column; so 'Dead' is now coded as -1 rather than 0
titanic.data[titanic.data$Survived == 0, "Survived"] <- -1
##There are some NAs in Age, fill with the median value
titanic.data$Age[is.na(titanic.data$Age)] = median(titanic.data$Age, na.rm=TRUE)

##BUILD MODEL
##Randomly choose 70% of the data set as training data
set.seed(39)
titanic.train.indices <- sample(1:nrow(titanic.data), 0.7*nrow(titanic.data), replace=F)
titanic.train <- titanic.data[titanic.train.indices,]
dim(titanic.train)
summary(titanic.train$Survived)
##Select the other 30% as the testing data
titanic.test <- titanic.data[-titanic.train.indices,]
dim(titanic.test)
summary(titanic.test$Survived)
##You could also do this
#random.rows.test <- setdiff(1:nrow(titanic.data),random.rows.train)
#titanic.test <- titanic.data[random.rows.test,]

##Fitting decision model on training set
titanic.bst.model <- bst(titanic.train[,2:8], titanic.train$Survived, learner = "tree")
#titanic.bst.model <- bst(titanic.train[,2:8], titanic.train$Survived, learner = "tree", control.tree=list(maxdepth=2))
print(titanic.bst.model)

##MODEL EVALUATION
##Predict test set outcomes and report probabilities
titanic.bst.predictions <- predict(titanic.bst.model, titanic.test, type="class")
##Create the confusion matrix
titanic.bst.confusion <- table(titanic.bst.predictions, titanic.test$Survived)
print(titanic.bst.confusion)
##Accuracy
titanic.bst.accuracy <- sum(diag(titanic.bst.confusion)) / sum(titanic.bst.confusion)
print(titanic.bst.accuracy)
##Precision
titanic.bst.precision <- titanic.bst.confusion[2,2] / sum(titanic.bst.confusion[2,])
print(titanic.bst.precision)
##Recall
titanic.bst.recall <- titanic.bst.confusion[2,2] / sum(titanic.bst.confusion[,2])
print(titanic.bst.recall)
##F1 score
titanic.bst.F1 <- 2 * titanic.bst.precision * titanic.bst.recall / (titanic.bst.precision + titanic.bst.recall)
print(titanic.bst.F1)



