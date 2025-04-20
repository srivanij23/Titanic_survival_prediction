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

| Metric            | Value     |
|------------------|-----------|
| Accuracy          | 1.0000    |
| Precision         | 1.0000    |
| Recall            | 1.0000    |
| F1 Score          | 1.0000    |
| Cross-Validation  | 1.0000 ± 0.0000 |

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
