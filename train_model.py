# train_model.py
# Matches the pipeline from the uploaded notebook:
# Car_Name, year, km_driven, fuel, seller_type, transmission, owner, seats, Engine (CC)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---- Load Dataset ----
ds = pd.read_csv("car_price.csv")   # <-- put your CSV in this same folder

# ---- Check for missing values (notebook showed none, but just in case) ----
for col in ds.columns:
    if ds[col].isnull().sum() > 0:
        if ds[col].dtype == "object":
            ds[col].fillna(ds[col].mode()[0], inplace=True)
        else:
            ds[col].fillna(ds[col].median(), inplace=True)

# ---- Label Encode categorical columns (5 separate encoders, as in notebook) ----
car_le = LabelEncoder()
fuel_le = LabelEncoder()
seller_le = LabelEncoder()
transmission_le = LabelEncoder()
owner_le = LabelEncoder()

ds["Car_Name"]     = car_le.fit_transform(ds["Car_Name"])
ds["fuel"]         = fuel_le.fit_transform(ds["fuel"])
ds["seller_type"]  = seller_le.fit_transform(ds["seller_type"])
ds["transmission"] = transmission_le.fit_transform(ds["transmission"])
ds["owner"]        = owner_le.fit_transform(ds["owner"])

# ---- Feature / Target split ----
x = ds.drop("selling_price", axis=1)
y = ds["selling_price"]

# ---- Scale features (StandardScaler, as in notebook) ----
sc = StandardScaler()
sc.fit(x)
x = pd.DataFrame(sc.transform(x), columns=x.columns)

# ---- Train-Test Split ----
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=50
)

# ---- Train Model ----
dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=7, random_state=50)
dt.fit(x_train, y_train)

# ---- Evaluate ----
train_score = dt.score(x_train, y_train) * 100
test_score = dt.score(x_test, y_test) * 100
y_pred = dt.predict(x_test)

print("Train Score :", train_score)
print("Test Score  :", test_score)
print("MSE         :", mean_squared_error(y_test, y_pred))
print("MAE         :", mean_absolute_error(y_test, y_pred))
print("R2 Score    :", r2_score(y_test, y_pred))

# ---- Save model + scaler + all encoders + feature column order ----
encoders = {
    "Car_Name": car_le,
    "fuel": fuel_le,
    "seller_type": seller_le,
    "transmission": transmission_le,
    "owner": owner_le,
}

joblib.dump(dt, "car_price_model.pkl")
joblib.dump(sc, "scaler.pkl")                       # <-- this was missing in the notebook
joblib.dump(encoders, "label_encoders.pkl")         # <-- fixed: all 5 encoders saved together
joblib.dump(list(x.columns), "feature_columns.pkl")

print("\n✅ Model, scaler, encoders, and feature columns saved successfully.")
