📊 Customer Churn Prediction Project

📌 Project Description

This project focuses on predicting customer churn using machine learning techniques. Customer churn refers to customers who stop using a company's service. By analyzing customer data, this model helps businesses identify customers who are likely to leave, enabling better retention strategies.

The project uses the Telco Customer Churn dataset and applies data preprocessing, feature encoding, handling class imbalance, and multiple machine learning models to achieve accurate predictions.


 ⚙️ Setup Instructions

 1️⃣ Clone the Repository

```bash
git clone https://github.com/vigneshkite/Customer_Churn_Prediction_project_claysys.git
cd Customer_Churn_Prediction_project_claysys
```

 2️⃣ Install Required Libraries

Make sure Python is installed (recommended: Python 3.8+)

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```


 ▶️ Usage Instructions

Run the main Python script:

```bash
python main.py
```

The script will:

* Load and preprocess the dataset
* Encode categorical variables
* Handle class imbalance using SMOTE
* Train multiple models
* Evaluate performance
* Save the best model and encoders
* Predict churn for sample input data


 📦 Dependencies / Prerequisites

* Python 3.x
* Libraries:

  * numpy
  * pandas
  * matplotlib
  * seaborn
  * scikit-learn
  * imbalanced-learn (SMOTE)
  * xgboost
  * pickle (built-in)


 🧠 Solution Approach

 1. Data Preprocessing

* Removed unnecessary column (`customerID`)
* Converted `TotalCharges` to numeric
* Handled missing values
* Converted target variable (`Churn`) into binary (0/1)

 2. Feature Engineering

* Applied **Label Encoding** to categorical variables
* Saved encoders using pickle for future predictions

 3. Handling Imbalanced Data

* Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance classes

 4. Model Training

Trained multiple models:

* Decision Tree
* Random Forest
* XGBoost

Used **5-fold Cross Validation** to compare performance.

 5. Model Selection

* Selected **Random Forest Classifier** based on best accuracy

 6. Evaluation Metrics

* Accuracy Score
* Confusion Matrix
* Classification Report

 7. Model Saving

* Saved trained model (`customer_churn_model.pkl`)
* Saved encoders (`encoders.pkl`)

 8. Prediction

* Accepts new customer data
* Applies same preprocessing
* Predicts churn and probability


 🔄 Regular Commits

This project follows a structured and incremental development process. Below is the commit history representing each step of the pipeline:

 📌 Project Initialization

* Add files via upload
* Main file created
* Import of modules

 📊 Data Loading & Exploration

* Load CSV to code
* Error on load data
* Data analysis and visualization

 🧹 Data Preprocessing

* Remove customer ID
* Convert string of TotalCharges into float
* Label encoding of churn column
* Identifying object columns
* Apply label encoding and store encoders
* Save encoders to a pickle file

 🔀 Data Splitting & Balancing

* Split training and test data
* Synthetic Minority Over-sampling Technique (SMOTE)

 🤖 Model Building

* Model training and cross-validation results

 📈 Evaluation & Deployment

* Evaluate on test data
* Save the trained model as a pickle file
* Load the saved model and feature names
* Prediction using input data

Each commit represents a meaningful step in the machine learning pipeline, ensuring:

* Better version control
* Easy debugging
* Clear project understanding



 📈 Future Improvements

* Hyperparameter tuning
* Deploy using Streamlit / Flask
* Add Power BI dashboard for visualization
* Use advanced models like LightGBM


 👨‍💻 Author

**Vignesh L**

GitHub: https://github.com/vigneshkite


## 📌 Dataset

Telco Customer Churn Dataset


⭐ If you found this project useful, consider giving it a star!
