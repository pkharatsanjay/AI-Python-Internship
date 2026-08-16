from sklearn.linear_model import LinearRegression
import numpy as np

hours = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
marks = np.array([20, 40, 60, 80, 100])

model = LinearRegression()
model.fit(hours, marks)

study_hours = 6
prediction = model.predict([[study_hours]])

print("Study Hours:", study_hours)
print("Predicted Marks:", prediction[0])
