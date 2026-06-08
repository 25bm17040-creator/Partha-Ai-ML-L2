import joblib as jb
import pandas as pd

model=jb.load('model.pkl')
x_scaler=jb.load('x_scaler.pkl')
y_scaler=jb.load('y_scaler.pkl')

new_input=pd.DataFrame([[18,33.770,1,1,0,2]],columns=['age','bmi','children','sex_labelled','smoker_labelled','region_labelled'])
new_input_scaled=x_scaler.transform(new_input)

predicted_charge =model.predict(new_input_scaled)
predicted_charge=y_scaler.inverse_transform(predicted_charge)
print("CHARGE PREDICTED WILL BE ",predicted_charge[0])