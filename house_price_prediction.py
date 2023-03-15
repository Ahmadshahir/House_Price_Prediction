import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the training data
train_data = pd.read_csv("train.csv")

# Select relevant features
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
X = train_data[features]
y = train_data["SalePrice"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=5)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Validate the model
y_val_pred = model.predict(X_val)
rmse = sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation RMSE: {rmse}")

# Load the test data
test_data = pd.read_csv("test.csv")
X_test = test_data[features]

# Handle missing data in the test set
X_test = X_test.apply(lambda x: x.fillna(x.mean()), axis=0)

# Make predictions on the test data
y_test_pred = model.predict(X_test)

# Save the predictions to a CSV file
output = pd.DataFrame({"Id": test_data["Id"], "SalePrice": y_test_pred})
output.to_csv("submission.csv", index=False)
















'''
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the training data
train_data = pd.read_csv("train.csv")

# Select relevant features
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
X = train_data[features]
y = train_data["SalePrice"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Validate the model
y_val_pred = model.predict(X_val)
rmse = sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation RMSE: {rmse}")

# Load the test data
test_data = pd.read_csv("test.csv")
X_test = test_data[features]

# Handle missing data in the test set
X_test.fillna(X_test.mean(), inplace=True)

# Make predictions on the test data
y_test_pred = model.predict(X_test)

# Save the predictions to a CSV file
output = pd.DataFrame({"Id": test_data["Id"], "SalePrice": y_test_pred})
output.to_csv("submission.csv", index=False)
'''