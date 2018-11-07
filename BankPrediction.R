# BankPrediction.R


# Project initialisation --------------------------------------------------

install.packages("caret")
library(caret)
install.packages("tidyverse")
library(tidyverse)
install.packages("purrr")
library(purrr)
install.pacakges("corrplot")
library(corrplot)
install.packages("VIM")
library(VIM)
install.packages("dummies")
library(dummies)
install.packages("rpart")
library(rpart)

bank <- read.csv("bank_traindata.csv")
bank_test <- read.csv("bank_testdata.csv")
bank_ans <- read.csv("bank_testans.csv")

# Data Exploration --------------------------------------------------------

head(bank)
sum(bank$y == "yes") / nrow(bank) # for reference in prediction validation

bank_NA <- bank
bank_NA$pdays <- ifelse(bank_NA$pdays == 999, NA, bank_NA$pdays)

# Bank_known will be the dataset for NA imputation.
bank_NA[bank_NA == "unknown"] = NA

map_dbl(bank_NA, ~sum(is.na(.))/nrow(bank_NA)) # percent of missing values per variable.
map_dbl(bank_NA, ~ length(unique(.))) # number of unique values per variable. 1 value entails that the variable does not provide any information for model training. 

ranked_bank <- map_dfc(bank_NA, rank)
corrplot::corrplot(cor(ranked_bank, method = "spearman"), method = "square", type = "lower")

rm(ranked_bank, bank_NA)

# Data pre-processing -----------------------------------------------------

bank <- subset(bank, select = c(-pdays, -duration))

# kNN imputation (Optional)

# bank$pdays <- ifelse(bank$pdays == 999, NA, bank$pdays)
# bank[bank == "unknown"] = NA
# 
# library(VIM)
# bank <- kNN(bank, variable = c("job", "marital", "education", "default", "housing", "loan"), imp_var = FALSE)
# 
# # For bank_test prediction
# bank_test$pdays <- ifelse(bank_test$pdays == 999, NA, bank_test$pdays)
# bank_test[bank_test == "unknown"] = NA
# bank_test <- subset(bank_test, select = c(-pdays, -duration))
# bank_test <- kNN(bank_test, variable = c("job", "marital", "education", "default", "housing", "loan"), imp_var = FALSE)

vars <- subset(bank, select = -y)

# Converting factor class variables into dummies for PCA
str(vars)
dummy_vars <- dummy.data.frame(vars, names = c("job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"), drop = FALSE)

# PCA ---------------------------------------------------------------------

components <- prcomp(dummy_vars, scale = T) 

# Scree plot
std_dev <- components$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)
plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

# Cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

# Modeling ----------------------------------------------------------------

bank_tr_predict <- data.frame(y = bank$y, components$x)

# We are interested in first 40 PCAs (from scree plot)
bank_tr_predict <- bank_tr_predict[,1:41] # 41 counting y col

control <- trainControl(method="cv", number=10) # K-fold Cross validation

# Run a model
fit.lda <- train(y ~ ., 
                 data = bank_tr_predict, 
                 method = "lda", 
                 metric = "Accuracy", 
                 trControl = control)
fit.cart <- train(y ~ ., 
                  data = bank_tr_predict, 
                  method = "rpart", 
                  metric = "Accuracy", 
                  trControl = control)
fit.knn <- train(y ~ ., 
                 data = bank_tr_predict, 
                 method = "knn", 
                 metric = "Accuracy", 
                 trControl = control)

# The SVM and Random Forest techniques are too complex for my computer.

(results <- resamples(list(lda = fit.lda, 
                           cart = fit.cart, 
                           knn = fit.knn)))
summary(results)


# Test Data Predictions ---------------------------------------------------

bank_test <- read.csv("bank_testdata.csv")

bank_test <- subset(bank_test, select = c(-pdays, -duration))
bank_test <- select(bank_test, -y)
levels(bank_test$default) <- c("yes", "no", "unknown")
test_dummy <- dummy.data.frame(bank_test, names = c("job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"), drop = FALSE)

# Run bank_test obs. through PCA
pred <- predict(components, newdata = test_dummy)
pred <- as.data.frame(pred)

# Select the first 40 components
pred <- pred[,1:40]

# Make prediction for bank_test
prediction <- predict(fit.cart, pred)
prediction <- as.data.frame(prediction)
bank_prediction <- cbind(bank_test, prediction)

sum(bank_prediction$prediction == "yes") /nrow(prediction)
sum(bank$y == "yes") / nrow(bank) # for reference in prediction validation

write.csv(bank_prediction, "BankPrediction")

# Test Prediction Validation ----------------------------------------------

head(bank_prediction)
head(bank_ans)
testing_df <- cbind(select(bank_prediction, prediction), select(bank_ans, y))
testing_df <- testing_df %>% mutate(correct = ifelse(prediction == y, 1, 0))

# How good were our predictions?
sum(testing_df$correct)/nrow(testing_df)
confusionMatrix(testing_df$prediction, testing_df$y)
