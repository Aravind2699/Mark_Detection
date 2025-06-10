# filepath: /student-mark-prediction/student-mark-prediction/scripts/mark_student.py
# 1. Import Libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 2. Load the Dataset
path = '../datasets/Student_Marks.csv'
df = pd.read_csv(path)

# 3. Prepare Features and Target
X = df[['number_courses', 'time_study']]  # Features
y = df['Marks']                           # Target variable

# 4. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Save the Trained Model
joblib.dump(model, '../models/student_mark_predictor.pkl')

# 7. Load the Model
loaded_model = joblib.load('../models/student_mark_predictor.pkl')

# 8. Predict with New Data
new_data = [[4, 10]]  # Example: 4 courses, 10 hours
predicted = loaded_model.predict(new_data)
print("Predicted Mark:", predicted[0])

# 9. Evaluate the Model
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))

# 10. Visualize the Data
plt.scatter(df['time_study'], df['Marks'])
plt.xlabel('Time Studied')
plt.ylabel('Final Mark')
plt.title('Time vs Marks')
plt.show()

# 11. Reusable Prediction Function
def predict_mark(courses, hours):
    model = joblib.load('../models/student_mark_predictor.pkl')
    return model.predict([[courses, hours]])[0]

# Example call
if __name__ == "__main__":
    result = predict_mark(6, 12)
    print("Predicted Final Mark:", result)