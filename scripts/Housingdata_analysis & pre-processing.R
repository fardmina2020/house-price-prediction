# Load required libraries
library(tidyverse)
library(corrplot)
library(caret)

# Load Housing Data set
housing_df <- read.csv("train.csv")


## Part 1a: Exploratory Data Analysis (EDA) 

### Data Structure

#Dimensions of the data set

dim(housing_df)          # rows and columns

#Check feature types

sapply(housing_df, class)

### Missing Values Overview

#Missing values in each column of the data set
colSums(is.na(housing_df))

###  Basic Data Cleaning

# Remove the ID column
housing_df$Id <- NULL

#Handling missing values

# Numeric → median
num_cols <- names(housing_df)[sapply(housing_df, is.numeric)]
housing_df[num_cols] <- lapply(housing_df[num_cols], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

# Categorical → most frequent level
cat_cols <- names(housing_df)[sapply(housing_df, is.character)]
housing_df[cat_cols] <- lapply(housing_df[cat_cols], function(x){
  x[is.na(x)] <- names(sort(table(x), decreasing = TRUE))[1]
  return(x)
})

### Visualize Distributions

#Distribution of the Target Variable: SalePrice
ggplot(housing_df, aes(SalePrice)) +
  geom_histogram(bins = 30,fill = "blue") +
  labs(title = "Distribution of SalePrice") +
  theme_minimal()

# correlation plot
corr_matrix <- cor(housing_df[num_cols])
corrplot(corr_matrix, method = "color", type = "lower", tl.cex = 0.7)

#Check Relationships with SalePrice - Top correlations with the target

sort(corr_matrix[,"SalePrice"], decreasing = TRUE)[1:15]

#Relation of Ground living area square feet with Sale Price
ggplot(housing_df, aes(GrLivArea, SalePrice)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm") +
  labs(title = "GrLivArea vs SalePrice", 
       x = "Ground living area sq ft" ,y= "Sale Price") +
  theme_minimal()

#Relation of Total square feet of basement area with Sale Price
ggplot(housing_df, aes(TotalBsmtSF, SalePrice)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm",col ="red") +
  labs(title = "TotalBsmtSF vs SalePrice", 
       x = "Total basement area sq ft" ,y= "Sale Price")+
  theme_minimal()

# Average Sale Price for each Heating quality and condition 
ggplot(housing_df,aes(HeatingQC ,SalePrice ,fill=HeatingQC)) +
  geom_boxplot() + labs(title = "Sale Price for Heating Quality & Condition") +
  theme_minimal()


### Summary of Key Characteristics

#- Dataset has 1460 rows and 80 features (79 predictors + SalePrice).
#- Contains both numeric (continuous, discrete) and categorical features. There are 37 numeric and 43 categorical variables
#- Several features have missing values, especially: LotFrontage, Alley, FireplaceQu, PoolQC, MiscFeature, Fence.
#- SalePrice is right-skewed.
#- Strongest correlations with SalePrice include OverallQual, GrLivArea, GarageCars, TotalBsmtSF, YearBuilt.



## Part 2: Feature Engineering 

#Perform Feature Engineering to prepare your data for your models; remember that you should adjust the feature engineering to address the peculiarities of your model. 

### 1. Feature Selection 

#Correlation with SalePrice

#correlation 
sort(corr_matrix[,"SalePrice"], decreasing = TRUE)[1:15]

#LASSO Feature Selection
library(glmnet)
x <- model.matrix(SalePrice ~ ., housing_df)[,-1]
y <- housing_df$SalePrice
lasso_model <- cv.glmnet(x, y, alpha = 1)
coeffs <- coef(lasso_model, s = "lambda.min")

coeff_matrix <- as.matrix(coeffs)
coeff_matrix <- coeff_matrix[rownames(coeff_matrix) != "(Intercept)", , drop = FALSE]
# Sort by absolute value to find the most influential features
sorted_indices <- order(abs(coeff_matrix[, 1]), decreasing = TRUE)
sorted_coeffs <- coeff_matrix[sorted_indices, , drop = FALSE]
coef_vars <- head(sorted_coeffs, 15)
coef_vars

#Random Forest Importance
library(randomForest)
rf_model <- randomForest(SalePrice ~ ., data=housing_df, importance=TRUE)
importance(rf_model)[1:15,]

#Selected Features Chosen Based on All 3 Techniques are:
#  OverallQual, GrLivArea, TotalBsmtSF, GarageCars, GarageArea, YearBuilt, 1stFlrSF,
#  FullBath, TotRmsAbvGrd, Neighborhood (categorical, high impact)

#These features consistently appear as strong predictors across correlation, LASSO coefficients, and Random Forest importance. They also have direct logical relation to house value (size, quality, age, amenities).


### 2. Feature Extraction 

library(caret)
preproc <- preProcess(housing_df[num_cols], method=c("center", "scale", "pca"))
pca_df <- predict(preproc, housing_df[num_cols])

head(pca_df)


### 3. Feature Creation 

#House Age
HouseAge <- housing_df$YrSold - housing_df$YearBuilt

#TotalArea
TotalArea <- housing_df$GrLivArea + housing_df$TotalBsmtSF

# Interaction Terms
Qual_Area <- housing_df$OverallQual * TotalArea


### 4. Handling Categorical Data 

#Convert categorical variables to numerical using appropriate encoding methods (e.g., label encoding, one-hot encoding). 
#What challenges did you face when transforming categorical data? 
  
library(fastDummies)
df_encoded <- dummy_cols(housing_df, remove_first_dummy = TRUE,
                         remove_selected_columns = TRUE)


### 5. Imputation of Missing Values 

#Handle missing values using techniques using median values for numeric columns and most frequent level for Categorical variables.
 
### 6. Normalization and Scaling
  
scaled_df <- as.data.frame(scale(df_encoded[num_cols[-37]]))


### 7. Handling Imbalanced Data 
  
#In our dataset Target Variable is Sale Price which is continuous so Regression does not have class imbalance.



## Export cleaned data - ready for applying models
write.csv(scaled_df,"Housingdata_cleaned.csv")


