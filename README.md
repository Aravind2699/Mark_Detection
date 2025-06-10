# Student Mark Prediction Application

This project is a student mark prediction application that utilizes machine learning to predict student marks based on the number of courses taken and the time spent studying. The application is built using Python and leverages libraries such as pandas, scikit-learn, joblib, and matplotlib.

## Project Structure

```
student-mark-prediction
├── datasets
│   └── Student_Marks.csv
├── models
│   └── student_mark_predictor.pkl
├── scripts
│   └── mark_student.py
├── requirements.txt
└── README.md
```

## Files Description

- **datasets/Student_Marks.csv**: Contains the dataset used for training the model, including features like the number of courses and time studied, along with the target variable, Marks.

- **models/student_mark_predictor.pkl**: The saved model after training. This file can be loaded to make predictions on new data.

- **scripts/mark_student.py**: This script is responsible for training the model and saving it. It includes data loading, preprocessing, model training, evaluation, and a reusable prediction function.

- **requirements.txt**: Lists the dependencies required for the project, such as pandas, scikit-learn, joblib, and matplotlib.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd student-mark-prediction
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the training script to train the model:
   ```
   python scripts/mark_student.py
   ```

## Usage

To make predictions using the trained model, you can call the `predict_mark` function from the `mark_student.py` script. For example:
```python
from scripts.mark_student import predict_mark

predicted_mark = predict_mark(6, 12)  # Example: 6 courses, 12 hours studied
print("Predicted Final Mark:", predicted_mark)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.