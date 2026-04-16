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
#Model Training
# dictionary of models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}
# dictionary to store the cross validation results
cv_scores = {}

# perform 5-fold cross validation for each model
for model_name, model in models.items():
  print(f"Training {model_name} with default parameters")
  scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="accuracy")
  cv_scores[model_name] = scores
  print(f"{model_name} cross-validation accuracy: {np.mean(scores):.2f}")
  print("-"*70)
  print(cv_scores)

  #RandomForest has high accuracy
  rfc = RandomForestClassifier(random_state=42)
  rfc.fit(X_train_smote, y_train_smote)
  print(y_test.value_counts())

  # evaluate on test data
y_test_pred = rfc.predict(X_test)

print("Accuracy Score:\n", accuracy_score(y_test, y_test_pred))
print("Confsuion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# save the trained model as a pickle file
model_data = {"model": rfc, "features_names": X.columns.tolist()}


with open("customer_churn_model.pkl", "wb") as f:
  pickle.dump(model_data, f)

# load the saved model and the feature names

with open("customer_churn_model.pkl", "rb") as f:
  model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]  
print(loaded_model)
print(feature_names)