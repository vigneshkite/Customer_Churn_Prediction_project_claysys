<h1>Customer Churn Prediction Project</h1> 

<h2>Project Description</h2>

This project focuses on predicting customer churn using machine learning techniques. Customer churn refers to customers who stop using a company's service. By analyzing customer data, this model helps businesses identify customers who are likely to leave, enabling better retention strategies.
The project uses the Telco Customer Churn dataset and applies data preprocessing, feature encoding, handling class imbalance, and multiple machine learning models to achieve accurate predictions.


<h2>Setup Instructions</h2>

 <h3>1️ Clone the Repository</h3>

```bash
git clone https://github.com/vigneshkite/Customer_Churn_Prediction_project_claysys.git
cd Customer_Churn_Prediction_project_claysys
```

 <h3>2️ Install Required Libraries</h3>

Make sure Python is installed (recommended: Python 3.8+)

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```


 <h3> 3 Usage Instructions</h3>

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


<h2> Dependencies / Prerequisites</h2>

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


<h2>Solution Approach</h2>

<h3> 1. Data Preprocessing</h3>

* Removed unnecessary column (`customerID`)
* Converted `TotalCharges` to numeric
* Handled missing values
* Converted target variable (`Churn`) into binary (0/1)

 <h3>2. Feature Engineering</h3>

* Applied **Label Encoding** to categorical variables
* Saved encoders using pickle for future predictions

 <h3>3. Handling Imbalanced Data</h3>

* Used SMOTE (Synthetic Minority Over-sampling Technique) to balance classes

 <h3>4. Model Training</h3>

Trained multiple models:

* Decision Tree
* Random Forest
* XGBoost

Used 5-fold Cross Validation to compare performance.

<h3> 5. Model Selection</h3>

* Selected Random Forest Classifier based on best accuracy

 <h3>6. Evaluation Metrics</h3>

* Accuracy Score
* Confusion Matrix
* Classification Report

 <h3>7. Model Saving</h3>

* Saved trained model (`customer_churn_model.pkl`)
* Saved encoders (`encoders.pkl`)

 <h3>8. Prediction</h3>

* Accepts new customer data
* Applies same preprocessing
* Predicts churn and probability


<h2>Regular Commits</h2>

This project follows a structured and incremental development process. Below is the commit history representing each step of the pipeline:

<h3>Project Initialization</h3>

* Add files via upload
* Main file created
* Import of modules

<h3>Data Loading & Exploration</h3>

* Load CSV to code
* Error on load data
* Data analysis and visualization

<h3>Data Preprocessing</h3>

* Remove customer ID
* Convert string of TotalCharges into float
* Label encoding of churn column
* Identifying object columns
* Apply label encoding and store encoders
* Save encoders to a pickle file

<h3>Data Splitting & Balancing</h3>

* Split training and test data
* Synthetic Minority Over-sampling Technique (SMOTE)

<h3>Model Building</h3>

* Model training and cross-validation results

<h2>Evaluation & Deployment</h2>

* Evaluate on test data
* Save the trained model as a pickle file
* Load the saved model and feature names
* Prediction using input data

Each commit represents a meaningful step in the machine learning pipeline, ensuring:

* Better version control
* Easy debugging
* Clear project understanding

<h2>Future Improvements</h2>

* Hyperparameter tuning
* Deploy using Streamlit / Flask
* Add Power BI dashboard for visualization
* Use advanced models like LightGBM

<h2>Author</h2>

Vignesh L

GitHub: https://github.com/vigneshkite


<h2>Dataset</h2>

Telco Customer Churn Dataset


