import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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
    model=Sequential()
    model.add(Dense(64,activation='relu',input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
    early_stopping=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
    model.fit(x_train,y_train,validation_split=0.2,epochs=100,batch_size=32,callbacks=[early_stopping],verbose=1)
    return model

def evaluate_model(model,x_test,y_test):
    y_pred=model.predict(x_test)
    for i in range(len(y_pred)):
        if y_pred[i]>=0.5:
            y_pred[i]=1
        else:
            y_pred[i]=0
    return accuracy_score(y_test,y_pred),classification_report(y_test,y_pred),confusion_matrix(y_test,y_pred),f1_score(y_test,y_pred) 

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
    