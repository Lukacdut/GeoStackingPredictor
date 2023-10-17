import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import shap
# Load the data
data = pd.read_csv('path_to_your_drilling_data.csv')
X, y = data[['x', 'y', 'z']], data['lithology']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Base models
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'DecisionTree': DecisionTreeClassifier(max_depth=5),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
}
# Train base models and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Performance:")
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("="*50)
# Stacking
base_learners = [
    ("dt", models['DecisionTree']),
    ("rf", models['RandomForest']),
    ("xgb", models['XGBoost']),
    ("gbdt", models['GradientBoosting']),
    ("knn", models['KNN'])
stack_clf = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
stack_clf.fit(X_train, y_train)
y_pred_stack = stack_clf.predict(X_test)
print("Stacking Classifier Performance:")
print("F1 Score:", f1_score(y_test, y_pred_stack, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_stack, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_stack, average='weighted'))
# SHAP explanations
def plot_shap_values(model, X_data):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_data)
    shap.summary_plot(shap_values, X_data)
# SHAP for base models
for name, model in models.items():
    print(f"SHAP values for {name}:")
    plot_shap_values(model, X_test)
# SHAP for Stacking Classifier
print("SHAP values for Stacking Classifier:")
plot_shap_values(stack_clf.final_estimator_, stack_clf.transform(X_test))

# 3D Visualization
def plot_3d(x, y, z, values, title):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(x, y, z, c=values, cmap='viridis')
    fig.colorbar(p, label='Lithology')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(title)
    plt.show()

# Visualize predictions from Stacking Classifier in 3D
x = X_test['x'].values
y = X_test['y'].values
z = X_test['z'].values
plot_3d(x, y, z, y_pred_stack, "3D Geological Model Visualization using Stacking Classifier")

