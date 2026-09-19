import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('insurance.csv')

print(df.head())
print(df.info())
print(df.describe())
print(df.shape)

print(df.isnull().sum())

df = pd.get_dummies(df, columns=['sex','smoker','region'], drop_first=True)
print(df.head())

x = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# PLOTTING
plt.xlabel("Actual")
plt.ylabel("Predicted")
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")
plt.show()

residuals = y_test - y_pred
plt.xlabel("Predicted")
plt.ylabel("Residuals")
sns.scatterplot(x=y_pred, y=residuals, hue=X_test['smoker_yes'])
plt.axhline(y=0, color="red")
plt.show()