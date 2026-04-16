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
df[df["TotalCharges"]==" "]
len(df[df["TotalCharges"]==" "])
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
df["TotalCharges"] = df["TotalCharges"].astype(float)
print(df.info())