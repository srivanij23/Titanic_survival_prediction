# 🚢 Titanic Survival Prediction

This project uses a machine learning model to predict whether a passenger survived the Titanic disaster. It includes proper data preprocessing, model training, evaluation, and visualization.

---

## 📌 Problem Statement

Predict the survival of Titanic passengers based on features such as age, sex, class, fare, and embarkation point.  
The goal is to develop a classification model with high accuracy and performance.

---

## 📂 Dataset

- File used: `tested.csv`
- Features:
  - `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`
  - Target: `Survived` (0 = No, 1 = Yes)

---

## 🧼 Preprocessing

- Handled missing values:
  - `Age` and `Fare`: Filled with mean
  - `Embarked`: Filled with mode ('S')
- Encoded categorical columns:
  - `Sex`, `Embarked` using Label Encoding
- Optional: Scaled numerical data for model efficiency

---

## 🤖 Model Used

- **Decision Tree Classifier**
  - `max_depth = 3`
  - `random_state = 42`

---

## 🧪 Evaluation Metrics
The model is evaluated using the following metrics:

  1.Accuracy: The percentage of correct predictions.

  2.Precision: The proportion of true positives out of all positive predictions.

  3.Recall: The proportion of true positives out of all actual positives.

  4.F1 Score: The harmonic mean of precision and recall.

  5.Confusion Matrix: A matrix to visualize the performance of the model.

| Metric            | Value     |
|------------------|-----------|
| Accuracy          | 1.0000    |
| Precision         | 1.0000    |
| Recall            | 1.0000    |
| F1 Score          | 1.0000    |
| Cross-Validation  | 1.0000 ± 0.0000 |

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        50
           1       1.00      1.00      1.00        34

    accuracy                           1.00        84
   macro avg       1.00      1.00      1.00        84
weighted avg       1.00      1.00      1.00        84

> ⚠️ Note: Extremely high performance may indicate data leakage or overfitting. Please validate with a different dataset if possible.

---

## 📊 Visualizations

- Decision Tree Visualization
- Confusion Matrix (Blues Colormap)

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/srivanij23/rvival-Prediction.git
cd Titanic-Survival-Prediction
```

---

## 2.     Install the required dependencies:
```bash
pip install -r requirements.txt
```
Tools & Libraries Used

    Python
    Pandas, NumPy
    Scikit-learn
    Matplotlib
    Jupyter Notebook (optional for exploration)

   👩‍💻 Author

Srivani Jadav
🔗 GitHub | LinkedIn
