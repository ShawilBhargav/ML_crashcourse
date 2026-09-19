import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error

# CONVERTING TO DF FROM OBJECT
housing = fetch_california_housing()
print(housing.keys())
print(housing.data)
print(housing.target)
print(housing.feature_names)

housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
print(housing_df)

housing_df['MedHouseVal'] = housing.target
print(housing_df.info())

# TRAIN/ TEST
X = housing_df.drop('MedHouseVal', axis=1)
Y = housing_df['MedHouseVal']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)

model = LinearRegression()
model.fit(X_train, Y_train)

# PREDICT & EVALUATION
Y_pred = model.predict(X_test)

mean_abs_error = mean_absolute_error(Y_test, Y_pred)
root_mean_sq_error = root_mean_squared_error(Y_test, Y_pred)
rootsq_score = r2_score(Y_test, Y_pred)

print(f"Mean Absolute: {mean_abs_error:.2f}")
print(f"Root Mean Square:{root_mean_sq_error:.2f}")
print(f"R2 Square: {rootsq_score:.2f}")