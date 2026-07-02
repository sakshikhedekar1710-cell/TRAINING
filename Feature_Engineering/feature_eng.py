import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer
)

df = sns.load_dataset("titanic")
print(df.head())

#Missing Value Imputation(Numerical)
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="median")

df["age"] = num_imputer.fit_transform(df[["age"]])

#Missing Value Imputation(categorical)
cat_imputer = SimpleImputer(strategy="most_frequent")

df["embarked"] = cat_imputer.fit_transform(df[["embarked"]]).ravel()

#Label Encoding
le = LabelEncoder()

df["sex"] = le.fit_transform(df["sex"])

#One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=["embarked"],
    drop_first=True
)

#Standard Scaling
scaler = StandardScaler()

df["age_scaled"] = scaler.fit_transform(df[["age"]])

#Min-Max Scaling
minmax = MinMaxScaler()

df["fare_minmax"] = minmax.fit_transform(df[["fare"]])