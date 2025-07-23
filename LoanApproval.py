import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Read dataset
df = pd.read_csv('loan_approval_dataset.csv')

print(df.columns.tolist())  
print(df.head())
print(df.info())
print(df.describe())


# Preprocess the data
df.drop('loan_id' , axis=1 , inplace=True) #It’s just an identifier — doesn’t help prediction.

# Clean before encoding
df.columns = df.columns.str.strip().str.lower()

# For categorical values debugging
print(df['self_employed'].unique())
print(df['education'].unique())
print(df['loan_status'].unique())

df['education'] = df['education'].str.strip()
df['self_employed'] = df['self_employed'].str.strip()
df['loan_status'] = df['loan_status'].str.strip()


# Encode your target
df['loan_status'] = df['loan_status'].map({'Approved':1 , 'Rejected':0})

# Encode categorical vars (you can use labelencoder too since it doesn't have more than two categories)
df['education'] = df['education'].map({'Graduate':1, 'Not Graduate':0})
df['self_employed'] = df['self_employed'].map({'Yes':1, 'No':0})




# Train, Test, Split
x = df.drop('loan_status', axis=1)
y = df['loan_status']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y) # stratify -> 	Maintain the same class ratio in train/test split

# Train a classification model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#plot confusion matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.savefig('CM_LogisticReg.png', format='png', dpi=300)
plt.show()

# Feature importance
coefficients = pd.DataFrame({
    'Feature' : x.columns,
    'Coefficient' : model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=coefficients, x='Coefficient', y='Feature', palette='Set3')
plt.title('Feature Importance - Logistic Regression')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Feature_LogisticReg.png', format='png', dpi=300)
plt.show()

####### Decision tree #######

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Intialize and train
tree_model =DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
tree_model.fit(x_train,y_train)

# Predict
y_pred_tree = tree_model.predict(x_test)

# Evaluate the model
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print("\nClassification Report:\n", classification_report(y_test, y_pred_tree))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))

import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix Visualization
cm_tree = confusion_matrix(y_test, y_pred_tree)
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.savefig('CM_tree.png', format='png', dpi=300)
plt.show()

# Tree Feature importance
tree_coefficients = pd.DataFrame({
    'Feature' : x.columns,
    'Coefficient' : tree_model.feature_importances_
}).sort_values(by='Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=tree_coefficients, x='Coefficient', y='Feature', palette='Set3')
plt.title('Feature Importance - Decision Tree')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Feature_tree.png', format='png', dpi=300)
plt.show()

# For risk of overfitting analysis (balanced result)
print("Tree Depth:", tree_model.get_depth())
print("Number of Leaves:", tree_model.get_n_leaves())
print("Train Accuracy:", model.score(x_train, y_train))
print("Test Accuracy:", model.score(x_test, y_test))

