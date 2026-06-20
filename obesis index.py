import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



#preprocessing the data
data = pd.read_csv("obdata.csv")
print(data.head())
print(data.isnull().sum())
data = data.drop_duplicates()
print(data.shape)

labelencoder = LabelEncoder()
data['Gender'] = labelencoder.fit_transform(data['Gender'])
data['SMOKE'] = labelencoder.fit_transform(data['SMOKE'])
data["CALC"] = labelencoder.fit_transform(data["CALC"])
data["FAVC"] = labelencoder.fit_transform(data["FAVC"])
data["SCC"] = labelencoder.fit_transform(data["SCC"])
data["family_history_with_overweight"] = labelencoder.fit_transform(data["family_history_with_overweight"])
data["CAEC"] = labelencoder.fit_transform(data["CAEC"])
data['MTRANS'] = labelencoder.fit_transform(data['MTRANS'])
data["NObeyesdad"] = labelencoder.fit_transform(data["NObeyesdad"])


x= data.drop(["NObeyesdad"], axis=1)
y= data[["NObeyesdad"]]


x_scaler = StandardScaler()

x_scaled = x_scaler.fit_transform(x)


x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2, random_state=42)


model = Sequential(
    [
        Dense(13, activation='relu',input_shape =(x_train.shape[1],)),
        Dense(13, activation='relu'),
        Dense(13, activation='relu'),
        Dense(1, activation='sigmoid')

    ]
)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train,epochs=100)


y_pred = model.predict(x_test)
error = mean_squared_error(y_test, y_pred)
rms = np.sqrt(error)
print("RMS Error:", rms)


joblib.dump(model, "model.pkl")
joblib.dump(x_scaler, "x_scaler.pkl")
print("Model and scalers saved successfully.")