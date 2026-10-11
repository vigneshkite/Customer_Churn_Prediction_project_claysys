FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout=300 --retries=10 -r requirements.txt

COPY app.py .
COPY customer_churn_model.pkl .
COPY encoders.pkl .
COPY WA_Fn-UseC_-Telco-Customer-Churn.csv .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
