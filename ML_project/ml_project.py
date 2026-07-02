from re import L
import pandas as pd 
import numpy as np 
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os 

# Category_encoders is needed for Target Encoding text

try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders not installed.Target Encoding will be unavaliable")
finally:
    print("It is installed")

def main():
    print("Loading DataSet")
    file_path = 'train.csv'

    if not os.path.exists(file_path):
        print(f"Error cannot find '{file_path}'")
        return
    df = pd.read_csv(file_path)
    print(f"Dataset Loaded Successfully.Rows:{df.shape[0]},Features:{df.shape[1]}\n")

    # Handling Missing Values
    print("Handling Missing Data")
    print("Artificially deleting some 'Hits' (H) data to demonstrate ")

    # Artifically create missing data for the lesson
    df.loc[0:25,'H']=np.nan
    print(df.head())
    #create an 'Imputer' that replaces missing values with the median
    imputer = SimpleImputer(strategy='median')

    #Apply the imputer to the column to fill the blanks
    df['H'] = imputer.fit_transform(df[["H"]])
    print(f"Imputation complete . 'Hits' (H) now has {df['H'].isnull().sum()} null values.\n")
    print(df.head())


    #Skewed Distributions
    print("Evaluting the skeweness of the Runs (R) distribution...")

    #Apply np.log1p (logarithm +1) to compare the distribution of numbers
    df['LogRuns'] = np.log1p(df['R'])
    print(f"Log Transformation applied. New skewness: {df['LogRuns'].skew():.2f} (closer to 0 is prefectly balanced)\n")
    print(df.head())

    df['Team_ID'] = ["Team_"+str(np.random.randint(1,150)) for _ in range(len(df))]

    if TargetEncoder is not None:
        print("Applying Target Encoder")

        encoder = TargetEncoder()

        df['Team_ID_Encoded'] = encoder.fit_transform(df['Team_ID'],df['W'])
        print(df.head())

    else:
        print("Category Encoders not installed")

    # Feature Selection

    features_to_test = ['R','HR','SO','SB']

    X_features = df[features_to_test].fillna(0)
    y_target = df['W']

    selector = SelectKBest(score_func=mutual_info_regression,k=2)
    selector.fit(X_features,y_target)
    winning_features = selector.get_support()
    best_features = X_features.columns[winning_features].tolist()

    print(best_features)

    #splitting data
    X=df[best_features]
    Y=df['W']
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    print(f"Training Data Size: {X_train.shape}")
    print(f"Testing Data Size: {X_test.shape}")

    #Training Model
    model=LinearRegression()
    model.fit(X_train,Y_train)

    predictions=model.predict(X_test)   
    print(predictions)

    #Compare model predictions to the actual answer
    actual_wins=Y_test.head(3).values
    predicted_wins=predictions[:3]

    for i in range(3):
        predicted=round(predicted_wins[i])
        actual=actual_wins[i]
        difference=abs(actual-predicted)
    
        print(f"Model Guessed: {predicted}")
        print(f"Real Answer: {actual}")
        print(f"Difference: {difference}")

if __name__=="__main__":
    main()