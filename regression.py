import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  
import matplotlib.pyplot as plt


df=pd.read_csv('height_weight.csv')

df = df.sort_values(by='Height')

X = df[['Height']].values
y = df['Weight'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on test set: {mse:.2f}")


new_height = np.array([[171]])
predicted_weight = model.predict(new_height)
print(f"Predicted weight for height {new_height[0][0]} cm is {predicted_weight[0]:.2f} kg")

plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='orange', label='Test data')
plt.scatter(new_height, predicted_weight, color='green', label='Prediction (175 cm)')

X_sorted = np.sort(X, axis=0)
plt.plot(X_sorted, model.predict(X_sorted), color='red', label='Regression line')

plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight Linear Regression with Train-Test Split')
plt.legend()
plt.show()
