import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix,f1_score
def preprocess_data(file_path):
    df=pd.read_csv(file_path)
    df.drop(['id'],axis=1,inplace=True)
    #Converting specific columns to numeric
    for col in ["pcv","wc","rc"]:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    #Handling missing values
    for col in df.select_dtypes(include=['object']).columns:
        df.fillna(df[col].mode()[0],inplace=True)

    for col in df.select_dtypes(include=['float64','int64']).columns:
        df.fillna(df[col].mean(),inplace=True)
    #mapping categorical valuyes to numerical values
    map={"yes":1,"no":0,
      "present":1,"notpresent":0,
      "abnormal":1,"normal":0,
      "good":1,"poor":0,
      "ckd":1,"notckd":0}

    df=df.map(lambda x:map.get(str(x).strip().lower(),x) if isinstance(x,str)else x)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
    df_reduced = df.drop(columns=to_drop)
    #Scaling
    x=df_reduced.drop(['classification'],axis=1)
    x=StandardScaler().fit_transform(x)
    y=df_reduced["classification"]
    #PCA
    pca=PCA(n_components=0.95)
    x_reduced=pca.fit_transform(x)
    return x_reduced,y

def split_data(x_reduced,y):
    x_train,x_test,y_train,y_test=train_test_split(x_reduced,y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test

def train_model(x_train,y_train):
    model=RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(x_train,y_train)
    return model

def evaluate_model(model,x_test,y_test):
    accuracy=model.score(x_test,y_test)
    y_pred=model.predict(x_test)
    return accuracy,classification_report(y_test,y_pred),confusion_matrix(y_test,y_pred),f1_score(y_test,y_pred)

if __name__ == "__main__":
    file_path=r"C:\projects\Capstone Project\Data\kidney_disease.csv"
    x_reduced,y=preprocess_data(file_path)
    x_train,x_test,y_train,y_test=split_data(x_reduced,y)
    model=train_model(x_train,y_train)
    accuracy,report,cm,f1=evaluate_model(model,x_test,y_test)
    print(f"Model Accuracy: {accuracy*100:.2f}%")
    print("Classification Report:\n",report)
    print("Confusion Matrix:\n",cm)
    print(f"F1 Score: {f1:.2f}")
    