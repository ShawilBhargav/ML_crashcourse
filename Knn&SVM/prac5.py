import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

wine_ds = pd.read_csv('WineQT.csv')
# print(wine_ds.shape)
# print(wine_ds.info())
# print(wine_ds.isnull().sum())
# print(wine_ds.duplicated().sum())

wine_ds = wine_ds.drop(columns=['Id'])

x = wine_ds.drop(columns='quality')
y = wine_ds['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_scaled, y_train)
y_pred = knn.predict(x_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted')*100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted')*100:.2f}%")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted')*100:.2f}%")
print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")

# SVM with Radial Basis Func
svm = SVC(kernel='rbf')
svm.fit(x_train_scaled, y_train)
y_pred = svm.predict(x_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted')*100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted')*100:.2f}%")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted')*100:.2f}%")
print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")

# SVM with Linear Func
svm = SVC(kernel='linear')
svm.fit(x_train_scaled, y_train)
y_pred = svm.predict(x_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted')*100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted')*100:.2f}%")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted')*100:.2f}%")
print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")