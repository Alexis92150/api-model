import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import ClassifierMixin
import joblib

def ingest_data(file_path: str):
    """
    Ingest data from a file path and return a pandas DataFrame
    """
    return pd.read_excel(file_path)

def clean_data(df: pd.DataFrame)-> pd.DataFrame:
    #suppression des lignes avec des valeurs manquantes
    #remplacer les valeurs non numériques par des valeurs numeriques
    df = df[['survived', 'pclass', 'sex', 'age']]
    df['sex'] = df['sex'].map({'female': 1,'male':0})
    df.dropna(inplace=True)
    return df

def train_model(df: pd.DataFrame) -> ClassifierMixin:
    #instatiate model
    model = KNeighborsClassifier(3)
    y = df['survived']
    X = df[['pclass','sex','age']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20 ,random_state=42)
    #train model
    #Evaluate model
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    print(model.score(X_test, y_test))
    return model


if __name__ == "__main__":
    df = ingest_data("train/titanic.xls")
    df = clean_data(df)
    model = train_model(df)
    joblib.dump(model, "model_titanic.joblib")
   

