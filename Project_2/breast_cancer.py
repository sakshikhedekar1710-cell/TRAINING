import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score,
    auc
)
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
#----------------------------------------------------------------------------------------------------

data=load_breast_cancer(as_frame=True)
df=data.frame
print(df.head())
X=df.drop("target",axis=1)
y=df["target"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)
#----------------------------------------------------------------------------------------------------

rf= RandomForestClassifier()
rf.fit(X_train,y_train)
rf_predict=rf.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test,rf_predict))
print("Precision Score:",precision_score(y_test,rf_predict))
print("recall:",recall_score(y_test,rf_predict))
print("f1 score:",f1_score(y_test,rf_predict) )
print("\nClassification Report:", classification_report(y_test,rf_predict))

cm1=confusion_matrix(
    y_test,
    rf_predict
)
disp1=ConfusionMatrixDisplay(confusion_matrix=cm1)
disp1.plot(cmap='Blues')
plt.title("Confusion Matrix for Random forest")
plt.show()

y_pred1=rf.predict(X_test)
y_prob1=rf.predict_proba(X_test)[:,1]
roc_score1=roc_auc_score(y_test,y_prob1)
print(f"ROC AUC Score for Random Forest: {roc_score1}")

fpr,tpr,threshols=roc_curve(y_test,y_prob1)
plt.figure(figsize=(7,4))
plt.plot(fpr,tpr,label=f"AUC= {roc_score1:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True positive Rate")
plt.title("ROC Curve for Random Forest")
plt.legend()
plt.show()

#------------------------------------------------------------------------------------
xgb=XGBClassifier()
xgb.fit(X_train,y_train)
xgb_predict=xgb.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test,xgb_predict))
print("Precision Score:", precision_score(y_test,xgb_predict))
print("recall:",recall_score(y_test,xgb_predict))
print("f1 score:",f1_score(y_test,xgb_predict) )
print("\nClassification Report:", classification_report(y_test,xgb_predict))

cm2=confusion_matrix(
    y_test,
    xgb_predict
)
disp2=ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot(cmap='Blues')
plt.title("Confusion Matrix for XGBoost")
plt.show()

y_pred2=xgb.predict(X_test)
y_prob2=xgb.predict_proba(X_test)[:,1]
roc_score2=roc_auc_score(y_test,y_prob2)
print(f"ROC AUC Score for XGBoost: {roc_score2}")

fpr,tpr,threshols=roc_curve(y_test,y_prob2)
plt.figure(figsize=(7,4))
plt.plot(fpr,tpr,label=f"AUC= {roc_score2:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True positive Rate")
plt.title("ROC Curve for XGBoost")
plt.legend()
plt.show()
#--------------------------------------------------------------------------------------------

lgb=LGBMClassifier()
lgb.fit(X_train,y_train)
lgb_predict=lgb.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test,lgb_predict))
print("Precision Score:", precision_score(y_test,xgb_predict))
print("recall:",recall_score(y_test,lgb_predict))
print("f1 score:",f1_score(y_test,lgb_predict) )
print("\nClassification Report:", classification_report(y_test,lgb_predict))

cm3=confusion_matrix(
    y_test,
    lgb_predict
)
disp3=ConfusionMatrixDisplay(confusion_matrix=cm3)
disp3.plot(cmap='Blues')
plt.title("Confusion Matrix for LightGBM")
plt.show()

y_pred3=xgb.predict(X_test)
y_prob3=xgb.predict_proba(X_test)[:,1]
roc_score3=roc_auc_score(y_test,y_prob3)
print(f"ROC AUC Score for LGBM: {roc_score3}")

fpr,tpr,threshols=roc_curve(y_test,y_prob3)
plt.figure(figsize=(7,4))
plt.plot(fpr,tpr,label=f"AUC= {roc_score3:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True positive Rate")
plt.title("ROC Curve for LGBM")
plt.legend()
plt.show()


#-------------------------------------------------------------------------------
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
    ],
    "Precision" :[
        precision_score(y_test,rf_predict),
        precision_score(y_test,xgb_predict),
        precision_score(y_test,lgb_predict)
    ],
    "Recall":[
        recall_score(y_test,rf_predict),
        recall_score(y_test,xgb_predict),
        recall_score(y_test,lgb_predict)
    ],
    "f1 score":[
        f1_score(y_test,rf_predict),
        f1_score(y_test,xgb_predict),
        f1_score(y_test,lgb_predict)

    ],
    "ROC Score":[
        roc_auc_score(y_test,y_prob1),
        roc_auc_score(y_test,y_prob2),
        roc_auc_score(y_test,y_prob3)
    ]
    

})
print(comparison)
#-----------------------------------------------------------------------------------------------------

### Baseline model (XGBoost) 
print("Baseline Accuracy:", accuracy_score(y_test,xgb_predict))

### Grid Search (Simple)
param_grid={
    'n_estimators':[50,100,150],
    'max_depth':[3,5,7],
    'learning_rate':[0.01,0.1]
}
grid=GridSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy'
)
grid.fit(X_train,y_train)
print("Best Parameters")
print(grid.best_params_)
best_model=grid.best_estimator_
prediction=best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,prediction))

###Random Search(Simple)
random_params= {
    'n_estimators':[50,100,150,200],
    'max_depth':[3,4,5,6,7],
    'learning_rate':[0.01,0.05,0.1,0.2]
}
random_search=RandomizedSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_distributions=random_params,
    n_iter=10,
    cv=3,
    random_state=42
)
random_search.fit(X_train,y_train)
print(random_search.best_params_)
best_random=random_search.best_estimator_
prediction=best_random.predict(X_test)
print("Accuracy:", accuracy_score(y_test,prediction))

### Optuna(Simple)
import optuna
def objective(trial):
    model=XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators",50,200),
        max_depth=trial.suggest_int("max_depth",3,7),
        learning_rate=trial.suggest_float("learning_rate",0.01,0.2),
        random_state=42
    )
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    return accuracy_score(y_test,prediction)
study=optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=20)
print(study.best_params)
best_model=XGBClassifier(
    **study.best_params,
    random_state=42
)
best_model.fit(X_train,y_train)
prediction=best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,prediction))

from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=42)
X_sample,y_sample=ros.fit_resample(X_train,y_train)

from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(random_state=42)
X_sample,y_sample=rus.fit_resample(X_train,y_train)

from imblearn .over_sampling import SMOTE
smote=SMOTE(random_state=42)
X_sample,y_sample=smote.fit_resample(X_train,y_train)
print(y_sample.value_counts())

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(class_weight="balanced")
model.fit(X_train,y_train)

#XGBoost
ratio=len(y_train[y_train==0])/len(y_train[y_train==1])
model=XGBClassifier(scale_pos_weights=ratio,random_state=42)
model.fit(X_train,y_train)

### Threshold tuning
prob=model.predict_proba(X_test)[:,1]
threshold=0.40
predictions=(prob>=threshold).astype(int)

for threshold in [0.5,0.4,0.2]:
    predictions=(prob>=threshold).astype(int)
    print("Threshold:",threshold)
    print(precision_score(y_test,prediction))


###summary plot
import shap
explainer=shap.TreeExplainer(model)
shap_values=explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X.columns
)

###Force plot
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0],
    feature_names=X.columns,
    matplotlib=True
)

import matplotlib.pyplot as plt
plt.show()