import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(filepath):
    
    #load dataset
    titanic_test= pd.read_csv('data/tested.csv')
    
    # Fill missing values
    titanic_test['Age'].fillna(titanic_test['Age'].mean(), inplace=True)
    titanic_test['Fare'].fillna(titanic_test['Fare'].mean(), inplace=True)
    titanic_test['Embarked'].fillna('S', inplace=True)
    
    #encode categorical variables
    le=LabelEncoder()
    titanic_test['Sex']=le.fit_transform(titanic_test['Sex'])
    titanic_test['Embarked']=le.fit_transform(titanic_test['Embarked'])
    
    #feature selection for model

    features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    X=titanic_test[features]
    y=titanic_test['Survived']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X,y

    