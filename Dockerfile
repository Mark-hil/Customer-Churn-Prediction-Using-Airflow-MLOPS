FROM apache/airflow:2.7.3

USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python packages
RUN pip install --no-cache-dir \
    pandas>=1.3.0 \
    scikit-learn>=1.0.0 \
    mlflow>=1.20.0 \
    joblib>=1.1.0 \
    xgboost>=1.5.0 \
    lightgbm>=3.3.0 \
    catboost>=1.0.0 \
    matplotlib>=3.4.0 \
    seaborn>=0.11.0 \
    pyarrow>=6.0.0 \
    pyyaml>=6.0.0 \
    loguru>=0.6.0 \
    numpy>=1.21.0 \
    python-dateutil>=2.8.2 \
    imblearn>=0.8.0
