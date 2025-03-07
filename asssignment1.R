# sph6004 assignment 1
# load required libraries
library(caret)  # For data splitting and validation
#library(randomForest) 
#library(rpart)  # Decision tree 
#library(rpart.plot)  # Plotting decision trees
library(ggplot2) 
library(dplyr)
library(corrplot) # Correlation plot
#library(glmnet)
#library(GA)
library(nnet)
library(MASS)
#library(gridExtra)
#load data
data = read.csv("/Users/cuprum/Desktop/SPH6004/sph6004_assignment1_data.csv")
#View(data)

#set seed for reproducibility
set.seed(13)

###################################### data cleaning
# check for missing values
sum(is.na(data))  # 3495921 NAs

col_missing <- colMeans(is.na(data))
row_missing <- rowMeans(is.na(data))
row_missing[1:20] #have an overview
col_missing[1:20]
range(col_missing[ 7:165])
#plot(col_missing)
#plot(row_missing)

par(mfrow = c(1, 2))
hist(row_missing,
     main = "Histogram of Row Missing", 
     xlab = "Row Missing", 
     col = "skyblue", 
     border = "black", 
     breaks = 10) 

hist(col_missing,
     main = "Histogram of Column Missing",
     xlab = "Column Missing",
     col = "salmon",
     border = "black",
     breaks = 10)

data_clean<- data[, col_missing < 0.5] #drop these col with too many NA
data_clean<- data_clean[row_missing < 0.6, ]
# remaining NA use mean to replace if <50%, 50-90% use binary value (if serious type then check some values)
sum(is.na(data_clean)) #
cm <- colMeans(is.na(data_clean))
data_clean <- data_clean %>%
  mutate(across(where(is.numeric), ~ replace(., is.na(.), mean(., na.rm = TRUE))))

#checking scaling of the column, remove obvious outliers
summary_stats <- summary(data_clean)
summary_stats #such as weight, there are some incorrect values

#remove rows with very abnormal outliers in numeric column using IQR * factor
filter_outliers <- function(df, multiplier) {
  #keep 7+ columns, remove gender race or id etc.
  cols <- names(df)[7:ncol(df)]
  keep <- rep(TRUE, nrow(df))
  
  for (col in cols) {
    q1 <- quantile(df[[col]], 0.1, na.rm = TRUE)
    q3 <- quantile(df[[col]], 0.9, na.rm = TRUE)
    iqr_val <- q3 - q1
    lower_bound <- q1 - multiplier * iqr_val
    upper_bound <- q3 + multiplier * iqr_val
    keep <- keep & (df[[col]] >= lower_bound & df[[col]] <= upper_bound)
  }
  filtered_df <- df[keep, ]
  removed_df <- df[!keep, ]
  return(list(filtered = filtered_df, removed = removed_df))
} #search online for outlier filter also

result_data <- filter_outliers(data_clean, multiplier = 3) # Values above Q3 + 3xIQR or below Q1 - 3xIQR are considered as extreme points
data_filtered <- result_data$filtered

#how many rows remain
dim(data_filtered)
summary(data_filtered) #notice there's some weight of 1kg, remove very small value such as these
data_filtered <- data_filtered[data_filtered$weight_admit >= 30, ] #arbitary set 30 for min
dim(data_filtered)
#check on removed data a bit
head(result_data$removed)


############################################# feature selection
# Simple check correlation
# correlation matrix using pairwise complete observations
cor_matrix <- cor(data_filtered[7:length(data_filtered[1,])], use = "pairwise.complete.obs")
# plot the matrix
par(mfrow = c(1, 1))
corrplot(cor_matrix, method = "color", order = "hclust", tl.pos = "n")

#identify highly correlated features (default 0.9)
highlyCorrelated <- findCorrelation(cor_matrix, cutoff = 0.8)
highlyCorrelated #take a look

# remove the highly correlated features
cols_to_remove <- (7:length(data_filtered[1,]))[highlyCorrelated]
data_reduced <- data_filtered[, -cols_to_remove]
#add a new column for binary kidney injury status
data_reduced$gender_encode <- as.numeric(as.factor(data_reduced$gender))
data_reduced$race_encode   <- as.numeric(as.factor(data_reduced$race))

data_reduced <- data_reduced[,c(-1,-2, -4, -6)]
data_reduced$aki_stage <- as.factor(data_reduced$aki_stage)

#by examinationn of the gcs score, it is either the sum of the remaining score, or = 15 when unable = 1 / verbal = 0
#thus removed the total score gcs_min to avoid issue
data_reduced <- subset(data_reduced, select = -gcs_min)

# Extract the target variable (first column)
target <- data_reduced[, 1]

# Extract features: columns 2 through (ncol - 2), skipping the last two columns
features <- data_reduced[, 2:(ncol(data_reduced) - 2)]

# Normalize the features using scale (mean = 0, sd = 1)
features_normalized <- as.data.frame(scale(features))
last_two <- data_reduced[, (ncol(data_reduced) - 1):ncol(data_reduced)]
data_reduced_normalized <- cbind(target, features_normalized, last_two)
colnames(data_reduced_normalized)[1] <- colnames(data_reduced)[1]
# Create boxplots for the normalized data

# Adjust the plotting layout and margins
par(mfrow = c(1, 2), mar = c(4, 4, 2, 1), oma = c(0, 0, 2, 0))

# First plot: Boxplot of the original data without x-axis labels
boxplot(features[, 2:11],
        main = "Boxplots before processing",
        col = "lightblue",
        las = 2,
        pch = 19,
        ylab = "Values",
        xaxt = "n")

# Second plot: Boxplot of normalized data without x-axis and y-axis labels
boxplot(data_reduced_normalized[, 2:11],
        main = "Boxplots after normalization",
        col = "lightblue",
        las = 2,
        pch = 19,
        ylab = "",
        xaxt = "n")


write.csv(data_reduced_normalized, "/Users/Cuprum/Desktop/SPH6004/sph6004_processed_data.csv", row.names = T)




#######################################################################################
# the following were models explored in the r session, while due to speed limitation
# the following has not been used
# python was used for analysis instead
#######################################################################################







X <- as.matrix(data_reduced)
y <- as.factor(data_filtered$aki_stage)
table(y)
# L1 regulization, the Lasso model using the best lambda
# Perform cross-validated Lasso (alpha = 1 indicates pure Lasso)
# Fit Lasso with cross-validation
#cv_lasso <- cv.glmnet(X, y, alpha = 1, family = "binomial")
#lambda_min <- cv_lasso$lambda.min    # minimizes CrossValidation error
#lambda_1se <- cv_lasso$lambda.1se    # sparser model

# Lasso model using lambda.min
#lasso_model_min <- glmnet(X, y, alpha = 1, lambda = lambda_min, family = "binomial")
#selected_features_min <- rownames(coef(lasso_model_min))[coef(lasso_model_min)[, 1] != 0]
#selected_features_min

# Lasso model using lambda.1se for fewer parameters
#lasso_model_1se <- glmnet(X, y, alpha = 1, lambda = lambda_1se, family = "binomial")
#selected_features_1se <- rownames(coef(lasso_model_1se))[coef(lasso_model_1se)[, 1] != 0]
#selected_features_1se

#L2 is not a feature selection, so here we use the combined Elastic Net Regression
# Load required libraries
library(glmnet)
library(MASS)

# --- Regularization and Feature Extraction ---

# Cross-validation for Elastic Net (using multinomial; change to binomial if binary)
cv_enet <- cv.glmnet(X, y, alpha = 0.5, family = "multinomial")
lambda_enet_1se <- cv_enet$lambda.1se

# Increase lambda further to force more coefficients to zero
lambda_tuned <- lambda_enet_1se * 2  # adjust multiplier as needed

# Fit the Elastic Net model with the tuned lambda
enet_model_tuned <- glmnet(X, y, alpha = 0.5, lambda = lambda_tuned, family = "multinomial")
enet_coef_tuned <- coef(enet_model_tuned)

# Extract feature names with nonzero coefficients from each class (ignoring the intercept)
selected_features_enet <- lapply(enet_coef_tuned, function(coefs) {
  features <- rownames(coefs)[coefs[, 1] != 0]
  setdiff(features, "(Intercept)")
})
all_selected_features <- unique(unlist(selected_features_enet))
print("Features selected by Elastic Net:")
print(all_selected_features)

###################### Stepwise Feature Selection ---

# Create a data subset using the outcome and the features selected above.
# It assumes that the names in 'all_selected_features' match the column names in data_reduced.
data_subset <- data_reduced #[, c("aki_stage", all_selected_features[-1])]
data_subset$aki_stage <- y

# Fit a full multinom model using all selected features
full_model <- multinom(aki_stage ~ ., data = data_subset, trace = FALSE)

# Backward Selection: Start with the full model and remove predictors iteratively
backward_model <- stepAIC(full_model, direction = "backward", trace = FALSE)
summary(backward_model)
selected_features_backward <- attr(terms(backward_model), "term.labels")

# Forward Selection: Start with a null model (only intercept) and add predictors step-by-step
null_model <- multinom(aki_stage ~ 1, data = data_subset, trace = FALSE)
forward_model <- stepAIC(null_model,
                         scope = list(lower = null_model, upper = full_model),
                         direction = "forward",
                         trace = FALSE)
summary(forward_model)
selected_features_forward <- attr(terms(forward_model), "term.labels")

# Find the column indices in data_reduced corresponding to these feature names
selected_features <- match(selected_features_backward, names(data_reduced))
#selected_features_enet  <- match(selected_names, names(data_reduced))

# select the features column
selected_columns <- selected_features[-1]

data_training <- data_reduced[,selected_columns]


########################preparing for the models
# Select features and target variable
X <- data_training
Y <- as.factor(data_filtered$aki_stage)

# Combine X and Y into a single dataframe for training
dataset <- data.frame(X, symptom = Y)

# Split into training (80%) and testing (20%) sets
train_index <- createDataPartition(Y, p = 0.8, list = FALSE)
train_data <- dataset[train_index, ]
test_data <- dataset[-train_index, ]


################################################### model 1: logistic regression - multiple levels
# multinomial logistic regression model using the training data
# Prepare training data: convert predictors to matrix and extract the outcome
x_train <- model.matrix(symptom ~ ., data = train_data)[, -1]  # remove intercept
y_train <- train_data$symptom

# Set the elastic net mixing parameter (alpha = 0.5 is a balanced elastic net)
alpha_value <- 0.5

# Cross-validation to select optimal lambda for multinomial regression
cv_fit <- cv.glmnet(x_train, y_train, family = "multinomial", alpha = alpha_value)

# Fit the final model with the optimal lambda
final_model <- glmnet(x_train, y_train, family = "multinomial",
                      alpha = alpha_value, lambda = cv_fit$lambda.min)

# Display model coefficients
print(coef(final_model))

# Prepare test data
x_test <- model.matrix(symptom ~ ., data = test_data)[, -1]

# Make predictions on the test set using the chosen lambda
predictions <- predict(final_model, newx = x_test, s = cv_fit$lambda.min, type = "class")

# Evaluate the model using a confusion matrix
conf_mat <- confusionMatrix(as.factor(predictions), as.factor(test_data$symptom))
print(conf_mat)


################################################### model 2: decision tree

# Train Decision Tree model
rpart_model <- rpart(symptom ~ ., data = train_data, method = "class")

#formula(rpart_model)
# Visualize the decision tree
rpart.plot(rpart_model, type = 2, extra = 104, fallen.leaves = TRUE, main = "Decision Tree")

# predictions using the Decision Tree
rpart_predictions <- predict(rpart_model, test_data, type = "class")

# Compute confusion matrix and accuracy for Decision Tree
rpart_conf_matrix <- confusionMatrix(rpart_predictions, test_data$symptom)
rpart_conf_matrix #0.42, and 1 is unused, so unbalanced?

#cross validation
train_control <- trainControl(method = "cv", number = 10)
grid <- expand.grid(cp = seq(0.0005, 0.005, by = 0.0005))

model_cv <- train(symptom ~ ., data = train_data, method = "rpart", 
                  trControl = train_control, tuneGrid = grid)
print(model_cv)

################################################### model 3: random forest
# the random forest with 200 trees

# Define 10-fold cross-validation control
train_control <- trainControl(method = "cv", number = 5)

# Define a tuning grid for mtry. Here we use your calculation:
# mtry = floor(sqrt(ncol(train_data) - 1))
rf_grid <- expand.grid(mtry = floor(sqrt(ncol(train_data) - 1)))

# Train the random forest model with cross-validation using caret's train function
rf_cv_model <- train(symptom ~ ., 
                     data = train_data, 
                     method = "rf", 
                     trControl = train_control, 
                     tuneGrid = rf_grid,
                     ntree = 500)

# Print the cross-validated model summary and best tuning parameter
print(rf_cv_model)

# Make predictions on the test set using the cross-validated model
rf_cv_predictions <- predict(rf_cv_model, newdata = test_data)

# Evaluate the model using a confusion matrix
conf_mat <- confusionMatrix(rf_cv_predictions, test_data$symptom)
print(conf_mat)


# Get feature importance metrics
importance_values <- importance(rf_cv_model$finalModel)
print(importance_values)

# Plot variable importance (Mean Decrease in Accuracy and Mean Decrease in Gini)
varImpPlot(rf_model)



