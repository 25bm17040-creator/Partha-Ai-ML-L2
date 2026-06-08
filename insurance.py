import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib as jb

data= pd.read_csv("insurance.csv")

label_encoder=LabelEncoder()
data["sex_labelled"]=label_encoder.fit_transform(data["sex"])
data["smoker_labelled"]=label_encoder.fit_transform(data["smoker"])
data["region_labelled"]=label_encoder.fit_transform(data["region"])

x= data.drop(columns=["sex","smoker","region","charges",])
y= data[["charges"]]

x_scaler=StandardScaler()
y_scaler=StandardScaler()

x_scaled=x_scaler.fit_transform(x)
y_scaled=y_scaler.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y_scaled,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_predict=model.predict(x_test)
error=mean_squared_error(y_test,y_predict)
performance=np.sqrt(error)
print("Estimated_performance in rms:",performance)

jb.dump(model,'model.pkl')
jb.dump(x_scaler,'x_scaler.pkl')
jb.dump(y_scaler,'y_scaler.pkl')
print("model and scaled files created")