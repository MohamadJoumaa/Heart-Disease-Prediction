import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""\
# Heart Disease Prediction
## 1. Dataset Collection & Preparation
This notebook loads the dataset, performs EDA, preprocesses the data, trains multiple models, evaluates them, and performs hyperparameter tuning.
"""))

cells.append(nbf.v4.new_code_cell("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
"""))

cells.append(nbf.v4.new_code_cell("""\
# Load the dataset
df = pd.read_csv('heart.csv')
display(df.head())
display(df.info())

# --- DATA CLEANING ---
print(f"Duplicates before cleaning: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

print(f"Missing values before cleaning:\\n{df.isnull().sum()}")
# Simple imputation for numerical/categorical or just drop
# Since it's a medical dataset, let's drop rows with missing critical values or fill them
df.dropna(inplace=True)

# Additional cleaning: e.g., Removing impossible values like zero resting BP or Cholesterol if that indicates missing data
# Some datasets use 0 as a missing value for Cholesterol or BP
df = df[df['RestingBP'] > 0]
# Depending on the dataset, Cholesterol = 0 might be missing data, but let's just do a basic drop
# df = df[df['Cholesterol'] > 0] 

print(f"Dataset shape after cleaning: {df.shape}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Exploratory Data Analysis
plt.figure(figsize=(10, 5))
sns.countplot(x='HeartDisease', data=df)
plt.title('Heart Disease Target Distribution')
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='Age', hue='HeartDisease', kde=True)
plt.title('Age Distribution by Heart Disease')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# Identify numerical and categorical columns
categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# Note: FastingBS is categorical (0 or 1)

# Create preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

trained_models = {}
for name, model in models.items():
    # Create pipeline combining preprocessor and model
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    # Train
    clf.fit(X_train, y_train)
    trained_models[name] = clf
    
    # Predict & Evaluate
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"--- {name} ---")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
"""))

cells.append(nbf.v4.new_code_cell("""\
# Tune Random Forest as it usually performs best
print("Tuning Random Forest...")

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation F1 Score: {grid_search.best_score_:.4f}")

# Update trained models with the tuned model
trained_models['Random Forest Tuned'] = grid_search.best_estimator_

# Evaluate tuned model on test set
y_pred_tuned = grid_search.best_estimator_.predict(X_test)
print("--- Random Forest Tuned ---")
print(f"Test F1 Score: {f1_score(y_test, y_pred_tuned):.4f}")
"""))


cells.append(nbf.v4.new_code_cell("""\
# Save all models for API deployment
joblib.dump(trained_models['Decision Tree'], 'dt_model.pkl')
joblib.dump(trained_models['Random Forest Tuned'], 'rf_model.pkl') # Save the tuned one
joblib.dump(trained_models['KNN'], 'knn_model.pkl')

print("Models saved successfully!")
"""))

nb['cells'] = cells
with open('notebook.ipynb', 'w') as f:
    nbf.write(nb, f)
print("notebook.ipynb generated successfully!")
