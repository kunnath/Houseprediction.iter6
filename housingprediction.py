import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import xgboost as xgb

# Load the dataset
path= './housing_iteration_6_regression1.csv'
df = pd.read_csv(path)
# Separate the 'Id' column
df_copy = df.copy()
id_col = df_copy.pop('Id')
df = df.drop(columns=['Id'])
# Define your target and features
target = 'SalePrice'
X = df.drop(columns=[target])
y = df[target]

# Identify numerical and categorical columns
numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
categorical_columns = X.select_dtypes(exclude=['number']).columns.tolist()


# Define the preprocessor for numerical and categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_columns),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_columns)
    ])

# Define the full pipeline with a RandomForestRegressor model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('selector', VarianceThreshold(threshold=0.1)),
    ('model', RandomForestRegressor(random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Hyperparameter tuning using GridSearchCV for Random Forest
rf_param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

rf_grid_search = GridSearchCV(pipeline, rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

# Best Random Forest model
best_rf_model = rf_grid_search.best_estimator_
rf_y_pred = best_rf_model.predict(X_test)

# Define the full pipeline with an XGBoost Regressor model
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('selector', VarianceThreshold(threshold=0.1)),
    ('model', xgb.XGBRegressor(random_state=42))
])

# Hyperparameter tuning using GridSearchCV for XGBoost
xgb_param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7, 10],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__subsample': [0.7, 0.8, 0.9, 1.0]
}

xgb_grid_search = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)

# Best XGBoost model
best_xgb_model = xgb_grid_search.best_estimator_
xgb_y_pred = best_xgb_model.predict(X_test)

# Evaluate models
def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1-r2) * (len(y_test)-1) / (len(y_test)-X_test.shape[1]-1)
    return mae, mse, rmse, r2, adjusted_r2

# Random Forest Evaluation
rf_mae, rf_mse, rf_rmse, rf_r2, rf_adjusted_r2 = evaluate_model(y_test, rf_y_pred)
print(f"Random Forest - MAE: {rf_mae}, MSE: {rf_mse}, RMSE: {rf_rmse}, R²: {rf_r2}, Adjusted R²: {rf_adjusted_r2}")

# XGBoost Evaluation
xgb_mae, xgb_mse, xgb_rmse, xgb_r2, xgb_adjusted_r2 = evaluate_model(y_test, xgb_y_pred)
print(f"XGBoost - MAE: {xgb_mae}, MSE: {xgb_mse}, RMSE: {xgb_rmse}, R²: {xgb_r2}, Adjusted R²: {xgb_adjusted_r2}")

# Best parameters and best score
print(f'Best Random Forest parameters: {rf_grid_search.best_params_}')
print(f'Best Random Forest score: {-rf_grid_search.best_score_}')
print(f'Best XGBoost parameters: {xgb_grid_search.best_params_}')
print(f'Best XGBoost score: {-xgb_grid_search.best_score_}')


# Create the submission DataFrame
submission_file = pd.DataFrame({
    'Id': id_col[X_test.index],  # Match the 'Id' column with the test set index
    'SalePrice': xgb_y_pred
})

# Save the DataFrame to a CSV file
csv_file_path = './test_example.csv'
submission_file.to_csv(csv_file_path, index=False)

print("Submission file saved successfully.")
