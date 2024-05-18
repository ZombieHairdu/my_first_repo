## Setup Environment

# Packages
library(data.table)
library(caret)
library(randomForest)
library(e1071)

## Load / Explore Data
test_dt <- fread('./data_in/test.csv')
train_dt <- fread('./data_in/train.csv')

head(train_dt)
summary(train_dt)
str(train_dt)  

# Data Cleaning
colSums(is.na(train_dt))
train_dt[is.na(Age),Age := mean(Age, na.rm = TRUE)]
train_dt[is.na(Embarked), Embarked := 'S']
train_dt[is.na(Age),Age := ceiling(mean(train_dt[!is.na(Age), Age]))]

# Feature Engineering
train_dt[,FamilySize := SibSp + Parch + 1]

# Convert Factors (categorical variables should be treated as factors)
train_dt[,`:=`(Survived = as.factor(Survived),
               Pclass = as.factor(Pclass))]

## Model Building

# Split the train set
set.seed(123)
trainIndex <- createDataPartition(train_dt[,Survived], p = .8, list = FALSE, times = 1)

trainData <- train_dt[trainIndex,]
testData <- train_dt[-trainIndex,]

# train the model
model <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + FamilySize,
                      data = trainData,
                      importance = TRUE,
                      ntree = 100)

# evaluate the model
predictions <- predict(model, testData)
confusionMatrix(predictions, testData[,Survived])

## Make Predictions
test_dt[is.na(Age),Age := mean(Age, na.rm = TRUE)]
test_dt[is.na(Embarked), Embarked := 'S']
test_dt[is.na(Age),Age := ceiling(mean(test_dt[!is.na(Age), Age]))]
test_dt[,FamilySize := SibSp + Parch + 1]
test_dt[,Pclass := as.factor(Pclass)]

# predict
test_dt[,survived := predict(model, test_dt)]

submission <- data.frame(PassengerId = test_dt[,PassengerId], Survived = test_dt[,survived])
write.csv(submission, file = './data_out/submission.csv', row.names = FALSE)
