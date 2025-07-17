import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Model paths
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/home/mark-hill/airflow-quickstart/data/models"))

# API settings
API_PREFIX = "/api/v1"
API_TITLE = "Churn Prediction API"
API_DESCRIPTION = "API for making churn predictions using trained models"
API_VERSION = "1.0.0"

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
