import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

heart_df = pd.read_csv("heart.csv")
# print(heart_df.shape)
# print(heart_df.dtypes)
# print(heart_df.isnull().sum())
# print(heart_df.duplicated().sum())
# print(heart_df.info())

slope_dict = {
    'Up' : 0,
    'Flat' : 1,
    'Down' : 2,
}
heart_df['ST_Slope'] = heart_df['ST_Slope'].map(slope_dict)
heart_df = pd.get_dummies(heart_df, columns=['Sex', 'ExerciseAngina', 'ChestPainType', 'RestingECG'], drop_first=True)

x = heart_df.drop(columns=['HeartDisease'])
y = heart_df['HeartDisease']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# DecisionTree (Unrestricted)
from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier()
dec_tree.fit(x_train, y_train)

dec_y_pred = dec_tree.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, dec_y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, dec_y_pred)*100:.2f}%")
print(f"Recall: {recall_score(y_test, dec_y_pred)*100:.2f}%")
print(f"F1: {f1_score(y_test, dec_y_pred)*100:.2f}%")
print(f"Confusion Matrix: {confusion_matrix(y_test, dec_y_pred)}")
print(dec_tree.get_depth())

# Decision Tree with limited depth
dec_tree_rest = DecisionTreeClassifier(max_depth=8)
dec_tree_rest.fit(x_train, y_train)

dec_y_pred_rest = dec_tree_rest.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, dec_y_pred_rest)*100:.2f}%")
print(f"Precision: {precision_score(y_test, dec_y_pred_rest)*100:.2f}")
print(f"Recall: {recall_score(y_test, dec_y_pred_rest)*100:.2f}")
print(f"F1: {f1_score(y_test, dec_y_pred_rest)*100:.2f}")
print(f"Confusion Matrix: {confusion_matrix(y_test, dec_y_pred_rest)}")

# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(x_train, y_train)

random_y_pred = random_forest.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, random_y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, random_y_pred)*100:.2f}")
print(f"Recall: {recall_score(y_test, random_y_pred)*100:.2f}")
print(f"F1: {f1_score(y_test, random_y_pred)*100:.2f}")
print(f"Confusion Matrix: {confusion_matrix(y_test, random_y_pred)}")

# importances of features
importances = random_forest.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': x.columns,
    'importance': importances
}).sort_values('importance', ascending=False)
print(feature_importance_df)

grid_params = {
    'n_estimators' : [50, 100, 200],
    'max_depth' : [2, 4, 8, None],
    'min_samples_split': [2, 5, 10]
}
grid_tuning = GridSearchCV(RandomForestClassifier(random_state=42), grid_params, cv=5, scoring='accuracy')
grid_tuning.fit(x_train, y_train)
print(f"Best param combi.: {grid_tuning.best_params_}")
print(f"Combi. cross-fold score: {grid_tuning.best_score_}")
print(f"Model fitted with best params: {grid_tuning.best_estimator_}")

best_model = grid_tuning.best_estimator_
y_pred_tuned = best_model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_tuned)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred_tuned)*100:.2f}")
print(f"Recall: {recall_score(y_test, y_pred_tuned)*100:.2f}")
print(f"F1: {f1_score(y_test, y_pred_tuned)*100:.2f}")
print(f"Confusion Matrix: {confusion_matrix(y_test, y_pred_tuned)}")