import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

#exploring data
df = pd.read_csv('StudentPerformanceFactors.csv')
df.head()
df.info()
df.describe()

#plot the relationship
plt.scatter(df['Hours_Studied'], df['Exam_Score'])
plt.title('Study Hours Vs Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.grid(True)
plt.savefig('ActualPlot.png', format='png', dpi=300)
plt.show()


#prepare the data
x = df[['Hours_Studied']]
y = df['Exam_Score']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#train the model
model = LinearRegression()
model.fit(x_train, y_train)

#evaluate the model (get prediction)
y_pred = model.predict(x_test)

#metrics
mse = mean_squared_error(y_test, y_pred)
rsquare = r2_score(y_test, y_pred)

print(f'Linear Regression MSE for: {mse:.2f}')
print(f'Linear Regression R² Score: {rsquare:.2f}')

#visualize predictions
plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Actual vs Predicted Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.legend()
plt.savefig('LinearPred.png', format='png', dpi=300)
plt.show()


########## Polynomial Regression #########


#create polynomial features
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

#split into train/test 
x_train_poly, x_test_poly, y_train, y_test = train_test_split(x_poly, y, test_size=0.2, random_state=42)

#train the polynomial model
poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)

#predict 
y_pred_poly = poly_model.predict(x_test_poly)

# Evaluate performance
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f'Polynomial Regression MSE: {mse_poly:.2f}')
print(f'Polynomial Regression R² Score: {r2_poly:.2f}')

# Generate a smooth curve for visualization
x_min = x['Hours_Studied'].min()
x_max = x['Hours_Studied'].max()
x_range = np.linspace(x_min, x_max, 200)
x_range_df = pd.DataFrame(x_range, columns=['Hours_Studied'])  # ✅ correct format
x_range_poly = poly.transform(x_range_df)
y_range_pred = poly_model.predict(x_range_poly)

# Plot actual points + polynomial curve
plt.scatter(x, y, color='blue', label='Actual')
plt.plot(x_range, y_range_pred, color='green', linewidth=2, label='Polynomial Fit')
plt.title('Polynomial Regression Fit')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()                                                        
plt.grid(True)
plt.savefig('PolyPred.png', format='png', dpi=300)
plt.show()


########### Multi Feature Regression ############


#multi regression
df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map({'Yes': 1, 'No': 0}) #You’re doing what's called label encoding

#Converts those categorical features into numerical columns
# Automatically encode ALL object (categorical) columns
df = pd.get_dummies(df, drop_first=True)


# Select multiple features (all except the exam score as it's the target var for y)
features = [col for col in df.columns if col not in ['Exam_Score']]


X = df[features]

# Target variable
y = df['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Multi Regression Using Feature Engineering MSE: {mse:.2f}')
print(f'Multi Regression Using Feature Engineering R² Score: {r2:.2f}')

#(Interpret Your Model)Print and analyze the feature importance (coefficients):
# Create a DataFrame from model coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False) 

# print(coefficients)was for debugging 


# Now let's plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color='skyblue')
plt.title('Feature Importance (Linear Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('FeatureImportance.png', format='png', dpi=300)
plt.show()




'''
this last solution is a multi feature regression but it has gave me no significant better results
BUT the surprise was using feature engineering and that's to use the categorical data after converting it to numeric and this by using dummies
so we have got all data columns in account and got too many features(except exam score)to enhance the performance evaluation
💡 By properly encoding categorical variables and selecting the right features, I've dramatically improved my model's ability to explain the data.
'''
