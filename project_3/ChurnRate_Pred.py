import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)


df = pd.read_csv("customer_churn_data.csv")
print("First 5 rows:")
print(df.head())

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nInformation")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

le=LabelEncoder()
df["Gender"]=le.fit_transform(df["Gender"])


df=df.drop(["Subscription Type","Contract Length"], axis=1)
print(df.head())
X=df.drop("Churn",axis=1)
y=df["Churn"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)

rf= RandomForestClassifier()
rf.fit(X_train,y_train)
rf_predict=rf.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test,rf_predict))
print("\nClassification Report:", classification_report(y_test,rf_predict))

xgb=XGBClassifier()
xgb.fit(X_train,y_train)
xgb_predict=xgb.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test,xgb_predict))
print("\nClassification Report:", classification_report(y_test,xgb_predict))

lgb=LGBMClassifier()
lgb.fit(X_train,y_train)
lgb_predict=lgb.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test,lgb_predict))
print("\nClassification Report:", classification_report(y_test,lgb_predict))

comparison=pd.DataFrame({
    "model":[
        "RandomForest",
        "XGBoost",
        "LightGBM"
    ],
    "Accuracy":[
        accuracy_score(y_test,rf_predict),
        accuracy_score(y_test,xgb_predict),
        accuracy_score(y_test,lgb_predict)
    ]

})
print(comparison)

important=pd.Series(
    rf.feature_importances_,
    index=X.columns
)
plt.figure(figsize=(8,6))
important.sort_values(
    ascending=False
).head(10).plot.barh()
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()