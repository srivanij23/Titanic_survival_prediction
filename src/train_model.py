from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.tree import DecisionTreeClassifier, plot_tree

def train_evalute(X,y):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#initialize and train model

    model=DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train,y_train)
     # Cross-validation to check consistency of accuracy
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validated accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
#make predictions
    y_pred=model.predict(X_test)
    
#evaluation metrics
#1.Accuracy
    accuracy=accuracy_score(y_test,y_pred)
    print(f"Accuracy:{accuracy:.4f}")

#2.precision
    precision=precision_score(y_test,y_pred)
    print(f"Precision: {precision:.4f}")

#3. Recall
    recall=recall_score(y_test,y_pred)
    print(f"Recall: {recall:.4f}")

#4. F1 Score
    f1=f1_score(y_test,y_pred)
    print(f"F1 Score: {f1:.4f}")
    
#5. Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test,y_pred))
    
#6. Confusion matrix
    cm=confusion_matrix(y_test,y_pred)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NotSurvived","Survived"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("confusion matrix")
    
#plot the decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=X.columns, class_names=["Not Survived", "Survived"])
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.show()


