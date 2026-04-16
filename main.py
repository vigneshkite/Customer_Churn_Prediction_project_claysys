import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

df = pd.read_csv("C:\\Users\\lvign\\OneDrive\\Desktop\\Customer Churn Prediction\\Customer_Churn_Prediction_project_claysys\\WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.shape)

print(df.head())
pd.set_option("display.max_columns", None)
print(df.head(2))
print(df.info())
df = df.drop(columns=["customerID"])
print(df.head(2))
# convert of total charges of string to float
df[df["TotalCharges"]==" "]
len(df[df["TotalCharges"]==" "])
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
df["TotalCharges"] = df["TotalCharges"].astype(float)
print(df.info())
df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
print(df.head(2))
#check of objects and selecting of object in dataset
object_columns = df.select_dtypes(include="object").columns
print(object_columns)
# initialize dictionary to save the encoders
encoders = {}
# apply label encoding and store the encoders
for column in object_columns:
  label_encoder = LabelEncoder()
  df[column] = label_encoder.fit_transform(df[column])
  encoders[column] = label_encoder
# save the encoders to a pickle file 
with open("encoders.pkl", "wb") as f:
  pickle.dump(encoders, f)

# splitting the features and target
X = df.drop(columns=["Churn"])
y = df["Churn"] 
# split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train.value_counts()) 
#Synthetic Minority Over-sampling Technique
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(y_train_smote.shape)
print(y_train_smote.value_counts())