# Part: Model Training and Comparison 

# Load cleaned dataset
housing_df <- read.csv("Housingdata_cleaned.csv")



## 1. Train Two Models 

# Feature selected by both feature selection methodology

# Recursive Feature Elimination (RFE)
features_rfe <- c(
  "GrLivArea","OverallQual","TotalBsmtSF","X1stFlrSF","GarageCars",
  "X2ndFlrSF","BsmtFinSF1","GarageArea","YearBuilt","MSZoning",
  "LotArea","ExterQual","Fireplaces","YearRemodAdd","GarageFinish",
  "OverallCond","CentralAir","MSSubClass","BsmtFinType1","Neighborhood"
)

# Stepwise Feature Selection
features_stepwise <- c(
  "OverallQual","GrLivArea","Neighborhood","BsmtFinSF1","MSSubClass",
  "OverallCond","YearBuilt","LotArea","TotalBsmtSF","Fireplaces",
  "MasVnrArea","GarageCars","YearRemodAdd","TotRmsAbvGrd"
)

# Train/Test Split - 80% train data and 20% test data

#set random seed
set.seed(123)
trainIndex <- createDataPartition(housing_df$SalePrice, p=0.8, list=FALSE)
train_data <- housing_df[trainIndex,]
test_data <- housing_df[-trainIndex,]


### Linear Regression Models


# Model using RFE features
formula_rfe <- as.formula(
  paste("SalePrice ~", paste(features_rfe, collapse="+"))
)
# Train Linear regression model
lm_rfe <- lm(formula_rfe, data=train_data)
# Prediction on test data
pred_lm_rfe <- predict(lm_rfe, test_data)

# Model using Stepwise features
formula_step <- as.formula(
  paste("SalePrice ~", paste(features_stepwise, collapse="+"))
)

# Train Linear regression model
lm_step <- lm(formula_step, data=train_data)
# Prediction on test data
pred_lm_step <- predict(lm_step, test_data)


### Random Forest Models

# Train Linear regression model - RFE features
rf_rfe <- randomForest(
  formula_rfe,
  data=train_data,
  ntree=200
)
# Prediction on test data
pred_rf_rfe <- predict(rf_rfe, test_data)

# Train Linear regression model - Stepwise features
rf_step <- randomForest(
  formula_step,
  data=train_data,
  ntree=200
)
# Prediction on test data
pred_rf_step <- predict(rf_step, test_data)


## 2. Evaluation and Compare

# Metrics for evaluation
metrics <- function(actual,pred){
  
  MAE <- mean(abs(actual-pred))
  MSE <- mean((actual-pred)^2)
  RMSE <- sqrt(MSE)
  R2 <- cor(actual,pred)^2
  
  return(c(MAE,MSE,RMSE,R2))
  
}

# Evaluation of each model
m1 <- metrics(test_data$SalePrice,pred_lm_rfe)
m2 <- metrics(test_data$SalePrice,pred_lm_step)
m3 <- metrics(test_data$SalePrice,pred_rf_rfe)
m4 <- metrics(test_data$SalePrice,pred_rf_step)

results <- data.frame(
  Model=c("LM-RFE","LM-STEP","RF-RFE","RF-STEP"),
  MAE=c(m1[1],m2[1],m3[1],m4[1]),
  MSE=c(m1[2],m2[2],m3[2],m4[2]),
  RMSE=c(m1[3],m2[3],m3[3],m4[3]),
  R2=c(m1[4],m2[4],m3[4],m4[4])
)

print(results)


## 3. Visualizations 

### Spider Chart

# 
radar_data <- results[,2:5]
radar_data <- rbind(apply(radar_data,2,max),
                    apply(radar_data,2,min),radar_data)

rownames(radar_data) <- c("max","min",results$Model)

radarchart(radar_data,axistype=1,title="Model Comparison Metrics")


### Residual Plots

# Linear Regression
res_lm_rfe  <- test_data$SalePrice - pred_lm_rfe
res_lm_step <- test_data$SalePrice - pred_lm_step

df_lm_res <- data.frame(
  Predicted = c(pred_lm_rfe, pred_lm_step),
  Residuals = c(res_lm_rfe, res_lm_step),
  Method = rep(c("RFE Features","Stepwise Features"), each = length(test_data$SalePrice)))

ggplot(df_lm_res, aes(x = Predicted, y = Residuals, color = Method)) + 
  geom_point(alpha = 0.6) + theme_bw() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residual Plot - Linear Regression",
       x = "Predicted SalePrice", y = "Residuals")

# Random Forest
res_rf_rfe  <- test_data$SalePrice - pred_rf_rfe
res_rf_step <- test_data$SalePrice - pred_rf_step

df_rf_res <- data.frame(
  Predicted = c(pred_rf_rfe, pred_rf_step),
  Residuals = c(res_rf_rfe, res_rf_step),
  Method = rep(c("RFE Features","Stepwise Features"), each = length(test_data$SalePrice))
)

ggplot(df_rf_res, aes(x = Predicted, y = Residuals, color = Method)) +
  geom_point(alpha = 0.6) + theme_bw() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residual Plot - Random Forest",
       x = "Predicted SalePrice", y = "Residuals")

### Predicted vs Actual Scatter Plot

# Linear Regression
df_lm <- data.frame(
  Actual = test_data$SalePrice,
  Predicted = c(pred_lm_rfe, pred_lm_step),
  Method = rep(c("RFE Features","Stepwise Features"), each=length(test_data$SalePrice)))

ggplot(df_lm, aes(x=Actual, y=Predicted, color=Method)) +
  geom_point(alpha=0.6) + theme_bw() +
  geom_abline(slope=1, intercept=0, linetype="dashed") +
  labs( title="Predicted vs Actual SalePrice (Linear Regression)",
        x="Actual SalePrice",y="Predicted SalePrice")

# Random Forest
df_rf <- data.frame(
  Actual = test_data$SalePrice,
  Predicted = c(pred_rf_rfe, pred_rf_step),
  Method = rep(c("RFE Features","Stepwise Features"), each=length(test_data$SalePrice))
)

ggplot(df_rf, aes(x=Actual, y=Predicted, color=Method)) +
  geom_point(alpha=0.6) + theme_bw() +
  geom_abline(slope=1, intercept=0, linetype="dashed") +
  labs( title="Predicted vs Actual SalePrice (Random Forest)",
        x="Actual SalePrice",y="Predicted SalePrice")
