#Predict the price of a used car using Kaggle used car dataset

import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import math
from pandas.api.types import is_numeric_dtype
df = pd.read_csv('C:/Users/Ethan Lapaczonek/Downloads/Used Car Dataset.csv')
print(df)
print(df.describe())
print(df.isnull().sum())
print(df[df["torque(Nm)"].isnull()])
df = df.dropna()
df = df.drop(columns=["Unnamed: 0"])
#Change price to USD

df["Price in USD"] = df["price(in lakhs)"].mul(1200.64)
print(df)
df = df.drop("price(in lakhs)", axis="columns")
print(df)
print(df.describe())
df = df.drop_duplicates()
nan_count = df.isna().sum()
print(nan_count)
#Add brand column and drop car_name
df['car_name']=df['car_name'].str.lower()
df['brand'] = df['car_name']
df["brand"] = df['brand'].str.split().str[1].tolist()
print(df["brand"])
print(df)
df = df.drop(["car_name", "registration_year"], axis="columns")
print(df)
print(df.describe())
#Fix outlier issues
print(df.groupby(["seats"]).count())
new_df = df[df["seats"]<=8]
print(new_df.describe())
print(new_df.groupby(["kms_driven"]).count())
new_df = new_df[new_df["kms_driven"]<260000]
print(new_df.describe())
new_df["number_of_miles"] = new_df["kms_driven"].mul(0.621371)
new_df = new_df.drop(["kms_driven"], axis=1)
print(new_df)
new_df["miles_per_gallon"] = new_df["mileage(kmpl)"].mul(2.35215)
new_df = new_df.drop(["mileage(kmpl)"], axis=1)
print(new_df.groupby(["miles_per_gallon"]).count())
new_df = new_df[new_df["miles_per_gallon"]<200]
print(new_df.groupby(["miles_per_gallon"]).count())
print(new_df.describe())
print(new_df.groupby(["torque(Nm)"]).count())
new_df = new_df[new_df["torque(Nm)"]<20000]
new_df['insurance_validity'] = new_df['insurance_validity'].replace('Third Party','Third Party insurance')
print(new_df.groupby(["max_power(bhp)"]).count())
new_df = new_df[new_df["max_power(bhp)"]<3000]
print(new_df.groupby(["Price in USD"]).count())
new_df = new_df[new_df["Price in USD"]<100000]
print(new_df)

print(new_df.dtypes)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
label_encoder = LabelEncoder()
x_categorical = new_df.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
print(x_categorical.corr())
#car name and year has high correlation, ownership and transmission is highly correlated
x_numerical = new_df.select_dtypes(exclude=['object'])
print(x_numerical.corr())
#max power and engine is correlated
x_numerical = x_numerical.drop(["max_power(bhp)"], axis="columns")
print(x_numerical.corr())

new_df = x_numerical.join(x_categorical)
print(new_df)




X = new_df.drop(['Price in USD'], axis=1)
y = new_df['Price in USD']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_error
rf = RandomForestRegressor(n_estimators=28, random_state=1)
model = rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(y_pred)
mse = mean_squared_error(y_test.values.ravel(), y_pred)
r2 = r2_score(y_test.values.ravel(), y_pred)
print('Mean Squared Error:', round(mse, 2))
print('R-squared scores:', round(r2, 2))
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', round(mae, 2))

from sklearn.ensemble import GradientBoostingRegressor
model_gb = GradientBoostingRegressor(n_estimators=7500, random_state=1, max_depth=5)
model_gb = model_gb.fit(X_train, y_train)
pred_gb = model_gb.predict(X_test)
mse_gb = mean_squared_error(y_test, pred_gb)
print(f"GRADIENT BOOSTED RANDOM FOREST REGRESSION IS : {mse_gb}")
r2_gb = r2_score(y_test, pred_gb)
print(f"GRADIENT R^2 IS: {r2_gb}")
mae_gb = mean_absolute_error(y_test, pred_gb)
print(f"GRADIENT MEAN SQUARED ERROR IS: {mae_gb}")

print(new_df)
print(df)
#Best model is Gradient boosted model:
#Now make it take input to predict the price of a car
car_predict = pd.DataFrame([[5, 3000, 305, 140000, 19, 0, 2, 1, 0, 2, 1]], columns=["seats","engine(cc)", "torque(Nm)", "number_of_miles", "miles_per_gallon", "insurance_validity", "fuel_type", "ownsership", "transmission", "manufacturing_year", "brand"])
print(car_predict)
print(X_test)
user_prediction = model_gb.predict(car_predict)
print(f"Predicted Price of the car is: {user_prediction}")
