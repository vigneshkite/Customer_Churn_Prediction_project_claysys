# 📊 Customer Churn Prediction

<div align="center">

  <h3>Machine Learning • Python • Streamlit • Docker</h3>

  <p>
    An interactive machine learning application that predicts customer churn
    and helps businesses identify customers who may be at risk of leaving.
  </p>

  <p>
    <a href="https://github.com/vigneshkite/Customer_Churn_Prediction_project_claysys">
      <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
    </a>
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11">
    <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  </p>

</div>

---

## 🚀 Project Overview

Customer churn is a major challenge for businesses that rely on customer retention. Losing customers can affect revenue, profitability, and long-term growth.

**Customer Churn Prediction** uses machine learning to estimate whether a customer is likely to leave a service. The project combines data preprocessing, a trained prediction model, and an interactive Streamlit interface.

<table>
  <tr>
    <td width="50%">
      <h3>🎯 Objective</h3>
      Identify customers who may be likely to churn using customer and service information.
    </td>
    <td width="50%">
      <h3>💡 Solution</h3>
      Provide an interactive application that uses a trained model to generate churn predictions.
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🧠 Machine Learning</h3>
      Process customer features and use saved model artifacts for prediction.
    </td>
    <td width="50%">
      <h3>🐳 Deployment</h3>
      Package the application using Docker for consistent execution across environments.
    </td>
  </tr>
</table>

## ✨ Key Features

- 📈 Customer churn prediction using a trained machine learning model
- 🖥️ Interactive web interface built with Streamlit
- 🧹 Data preprocessing and categorical feature encoding
- 🤖 Reusable trained model and encoder artifacts
- 📊 Customer data analysis and model training workflow
- 🐳 Docker-based application packaging
- ☁️ Ready for further cloud deployment

## 🛠️ Tech Stack

<div align="center">

| Technology | Purpose |
|:---:|---|
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="25" alt="Python"> **Python** | Application and model development |
| **Pandas & NumPy** | Data processing and analysis |
| **Scikit-learn** | Machine learning and preprocessing |
| **Streamlit** | Interactive web application |
| **Matplotlib / Seaborn / Plotly** | Data visualization, where used |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg" width="25" alt="Docker"> **Docker** | Containerization |
| <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" width="25" alt="Git"> **Git & GitHub** | Version control and source code hosting |

</div>

## 🧠 Machine Learning Workflow

<table>
  <tr>
    <td align="center" width="25%">
      <h3>01</h3>
      <b>Data Collection</b>
      <p>Load the Telco Customer Churn dataset.</p>
    </td>
    <td align="center" width="25%">
      <h3>02</h3>
      <b>Preprocessing</b>
      <p>Clean data and prepare features.</p>
    </td>
    <td align="center" width="25%">
      <h3>03</h3>
      <b>Model Training</b>
      <p>Train and evaluate candidate models.</p>
    </td>
    <td align="center" width="25%">
      <h3>04</h3>
      <b>Prediction</b>
      <p>Generate predictions through Streamlit.</p>
    </td>
  </tr>
</table>

## 📂 Project Structure

```text
Customer_Churn_Prediction_project_claysys/
│
├── app.py
├── data_analysis_model_training.py
├── customer_churn_model.pkl
├── encoders.pkl
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.11
- Git
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/vigneshkite/Customer_Churn_Prediction_project_claysys.git

cd Customer_Churn_Prediction_project_claysys
```

### 2. Create a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**WSL Ubuntu / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open the local application in your browser:

**http://localhost:8501**

## 🐳 Run with Docker

Make sure Docker Desktop or another compatible Docker engine is running.

### Build the Docker Image

```bash
docker build -t customer-churn-prediction .
```

### Start the Container

```bash
docker run -d --name churn-app -p 8501:8501 customer-churn-prediction
```

### Access the Application

Open **http://localhost:8501**.

### Useful Docker Commands

```bash
# View running containers
docker ps

# View application logs
docker logs churn-app

# Stop the application
docker stop churn-app

# Start the existing container again
docker start churn-app
```

## 📊 Dataset & Model Artifacts

**Dataset:** Telco Customer Churn

The dataset contains customer demographics, subscribed services, account details, billing information, and churn labels.

| File | Description |
|---|---|
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Customer churn dataset |
| `customer_churn_model.pkl` | Saved trained machine learning model |
| `encoders.pkl` | Saved categorical encoders |
| `data_analysis_model_training.py` | Data analysis and model training workflow |
| `app.py` | Streamlit application |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Docker image configuration |

> **Security:** Only load Pickle files from trusted sources. Do not commit API keys, passwords, access tokens, or other secrets to GitHub.

## 🔮 Future Improvements

- [ ] Deploy the application to AWS
- [ ] Automate Docker builds using GitHub Actions
- [ ] Add automated model evaluation reports
- [ ] Improve input validation and error handling
- [ ] Add model monitoring and data drift detection

## 👨‍💻 Author

<div align="center">

  <h3>Vignesh L</h3>
  <p>B.Tech — Artificial Intelligence & Data Science</p>

  <a href="https://github.com/vigneshkite">
    <img src="https://img.shields.io/badge/GitHub-vigneshkite-181717?style=for-the-badge&logo=github" alt="GitHub Profile">
  </a>

  <p><i>Building practical machine learning applications with Python.</i></p>

</div>

---

<div align="center">
  <p>⭐ If you find this project useful, consider starring the repository!</p>
  <sub>Built with Python, Machine Learning, and Streamlit.</sub>
</div>
