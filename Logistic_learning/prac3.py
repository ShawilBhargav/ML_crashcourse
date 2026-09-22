import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

bank_df = pd.read_csv('Customer-Churn-Records.csv')
# print(bank_df.info())
# print(bank_df.describe())
# print(bank_df.isnull().sum())

# print(bank_df.columns)
# print(bank_df['Exited'].value_counts())

car_rank = {
    'SILVER' : 0,
    'GOLD' : 1,
    'DIAMOND' : 2,
    'PLATINUM' : 3
}
bank_df['Card Type'] = bank_df['Card Type'].map(car_rank)
bank_df = bank_df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Complain'])
bank_df = pd.get_dummies(bank_df, columns=['Geography', 'Gender'])
# print(bank_df.columns)

x = bank_df.drop(columns=['Exited'])
y = bank_df['Exited']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1: {f1_score(y_test, y_pred)*100:.2f}%")
print(f"Confusion Matrix: {confusion_matrix(y_test, y_pred)}")

# Increasing recall by lowering the threshold 0.5 -> 0.3
prob_pred = model.predict_proba(x_test_scaled)
churn_pred = prob_pred[:, 1]
y_pred_new = churn_pred > 0.3
print(f"Accuracy: {accuracy_score(y_test, y_pred_new)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred_new)*100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred_new)*100:.2f}%")
print(f"F1: {f1_score(y_test, y_pred_new)*100:.2f}%")
print(f"Confusion Matrix: {confusion_matrix(y_test, y_pred_new)}")